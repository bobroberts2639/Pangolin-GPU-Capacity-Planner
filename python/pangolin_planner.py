"""
Pangolin GPU Planner – Python sizing engine
===========================================

A faithful Python port of the web app's sizing engine (``js/app.js``: ``size()``,
``costEst()`` and ``computeAll()``). It reads the **same** data file as the web app
(``js/data.js``), so both always use identical models, GPUs and assumptions.

Inputs use the same dictionary format as the web app's scenarios, so a
``pangolin-gpu-planner-scenarios.json`` file exported from the browser can be
loaded with :func:`load_scenarios` and sized here.

Quick start::

    import pangolin_planner as pp
    inp = pp.default_inputs(model="llama31-8b", gpu="h100", users=500)
    result = pp.compute_all(inp)
    print(result["best"]["gpu"]["name"], result["best"]["gpus"])
    pp.comparison_table(result)        # pandas DataFrame of all GPUs

Cost figures are ROUGH, INDICATIVE RANGES ONLY – see ``COST_CAVEAT``.
"""
from __future__ import annotations

import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

__version__ = "2.0.0"

INF = float("inf")
COST_CAVEAT = (
    "CAVEAT EMPTOR: cost figures are rough indicative ranges for early budgeting and relative "
    "comparison only. They are not quotes and may be wrong by ±30–50% or more. GPU, server, network "
    "and energy prices change quickly and vary with supplier, volume, contract terms, region, "
    "configuration and availability. Excluded: facility build-out, cooling plant, software licences, "
    "staffing, taxes, financing."
)

_DEFAULT_DATA = Path(__file__).resolve().parent.parent / "js" / "data.js"
_MARKER = "window.PLANNER_DATA ="


# --------------------------------------------------------------------------- data
def load_data(path: str | Path | None = None) -> dict:
    """Load the shared data file (``js/data.js``) used by the web app."""
    text = Path(path or _DEFAULT_DATA).read_text(encoding="utf-8")
    start = text.index("\n" + _MARKER) + 1 + len(_MARKER)   # the marker at the start of a line
    return json.loads(text[start: text.rindex(";")])


DATA = load_data()


def _model(mid: str, data: dict = None) -> dict:
    data = data or DATA
    for m in data["models"]:
        if m["id"] == mid:
            return m
    raise KeyError(f"Unknown model id '{mid}'. Valid ids: {[m['id'] for m in data['models']]}")


def _gpu(gid: str, data: dict = None) -> dict:
    data = data or DATA
    for g in data["gpus"]:
        if g["id"] == gid:
            return g
    raise KeyError(f"Unknown GPU id '{gid}'. Valid ids: {[g['id'] for g in data['gpus']]}")


def default_inputs(**overrides: Any) -> dict:
    """Return an inputs dict with the web app's defaults, optionally overridden.

    Keys: model, ov (dict of total/active/kv/state overrides), prec (FP32/FP16/FP8/FP4),
    usecase, isl, osl, users, tps, peak, gpu, util, gps, redund (none/n1/n2/p10/p20/2n),
    rankBy (servers/power/cost/gpus), cost (dict of kwh/years/util/maint/plo/phi/gpu).
    """
    inp = {
        "model": "llama31-8b", "ov": {}, "prec": "FP16", "usecase": "custom",
        "isl": 1024, "osl": 1024, "users": 100, "tps": 20, "peak": 1,
        "gpu": "h100", "util": 100, "gps": 2, "redund": "none", "rankBy": "servers",
        "cost": {"kwh": 0.12, "years": 3, "util": 50, "maint": 10},
    }
    cost = overrides.pop("cost", None)
    inp.update(overrides)
    if cost:
        inp["cost"] = {**inp["cost"], **cost}
    return inp


def load_scenarios(path: str | Path) -> list[dict]:
    """Load scenarios exported from the web app (``Export scenarios (.json)``)."""
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    return d["scenarios"] if isinstance(d, dict) else d


def model_from(inp: dict, data: dict = None) -> dict:
    """Built-in model record with any overrides from ``inp['ov']`` applied."""
    base = _model(inp["model"], data)
    m = dict(base)
    o = inp.get("ov") or {}
    if (o.get("total") or 0) > 0:
        m["total"] = o["total"]
    if (o.get("active") or 0) > 0:
        m["active"] = min(o["active"], m["total"])
    if o.get("kv") is not None and o["kv"] >= 0:
        m["kv"] = o["kv"]
    if o.get("state") is not None and o["state"] >= 0:
        m["state"] = o["state"]
    m["edited"] = any((o.get(k) or 0) > 0 and abs(o[k] - base[k]) > 1e-9 for k in ("total", "active", "kv")) or (
        o.get("state") is not None and o["state"] >= 0 and abs(o["state"] - (base.get("state") or 0)) > 1e-9)
    return m


# --------------------------------------------------------------------------- engine
def _next_pow2(n: float) -> int:
    p = 1
    while p < n:
        p *= 2
    return p


def _ram_step(gb: float, data: dict) -> float:
    for x in data["ramSteps"]:
        if x >= gb:
            return x
    return math.ceil(gb / 4096) * 4096


def tflops_for(g: dict, prec: str) -> float:
    if prec == "FP32":
        return g["fp32"]
    if prec == "FP8":
        return g["fp8"]
    if prec == "FP4":
        return g.get("fp4") or (g["fp8"] if g.get("fp8native") else g["fp16"])
    return g["fp16"]


def _infeasible(res: dict, why: str) -> dict:
    res.update(dict(infeasible=True, limit=why, gpus=INF, baseGpus=INF, spareGpus=0, replicas=math.nan,
                    servers=INF, facKW=INF, gpuKW=INF, itKW=INF, aggTps=math.nan, ttft=math.nan,
                    memUsedGB=math.nan, memAvailGB=math.nan, fitsServer=False, totalGpuMemRaw=math.nan,
                    totalTflops=math.nan, hostRecGB=math.nan, hostMinGB=math.nan))
    res.setdefault("perUserTps", math.nan)
    return res


def size(m: dict, g: dict, inp0: dict, data: dict = None) -> dict:
    """Size one model/workload on one GPU type. Mirrors ``size()`` in js/app.js."""
    data = data or DATA
    C = data["constants"]
    max_ps = data["maxGpusPerServer"].get(g["form"], 8)
    inp = {**inp0, "gps": max(1, min(inp0.get("gps") or 1, max_ps))}
    users = math.ceil(inp["users"] * (inp.get("peak") or 1))
    prec = inp["prec"]
    bpp = data["bytesPerParam"][prec]
    tflops = tflops_for(g, prec)
    usable = g["mem"] * inp["util"] / 100
    weights = m["total"] * bpp
    active_gb = m["active"] * bpp
    kv_tok_b = m["kv"] * 1024 * (data["kvBytes"][prec] / 2)
    ctx = inp["isl"] + inp["osl"]
    state_gb = (m.get("state") or 0) / 1000
    kv_user = kv_tok_b * ctx / 1e9 + state_gb
    kv_total = kv_user * users
    tps = inp.get("tps") or 0
    res = dict(gpu=g, gps=inp["gps"], gpsCapped=inp["gps"] < (inp0.get("gps") or 1), users=users, tflops=tflops,
               usable=usable, weightsGB=weights, activeGB=active_gb, kvTokB=kv_tok_b, stateGB=state_gb,
               kvUserGB=kv_user, kvTotalGB=kv_total, ctx=ctx, infeasible=False)
    if usable <= 0:
        return _infeasible(res, "Memory utilisation is 0%")

    ovh = 1 + C["overhead"]
    rep = _next_pow2(math.ceil((weights * ovh + kv_user * ovh) / usable))

    def speed_at1(r):
        return min(r * g["bw"] * C["bwEff"] / (active_gb + kv_user),
                   r * tflops * 1e12 * C["mfuDecode"] / (2 * m["active"] * 1e9))

    if tps > 0:
        while speed_at1(rep) < tps and rep < 64:
            rep *= 2
    if rep > inp["gps"] and inp["gps"] > 0:
        rep = math.ceil(rep / inp["gps"]) * inp["gps"]
    res["rep"] = rep
    if tps > 0 and speed_at1(rep) < tps:
        res["perUserTps"] = speed_at1(rep)
        return _infeasible(res, f"Cannot reach {tps} tok/s per user")

    frac = m["active"] / m["total"]
    mem_need_total = kv_total * ovh
    comp_need = users * tps * 2 * m["active"] * 1e9 / (tflops * 1e12 * C["mfuDecode"]) if tps > 0 else 0

    def solve(rp):
        c = comp_need / rp
        R = max(1, math.floor(min(mem_need_total / (rp * usable), c if c else INF) * 0.5))
        limit = "Memory capacity"
        while R <= users:
            N, b = R * rp, users / R
            ok = True
            if rp * usable < weights * ovh + kv_user * b * ovh:
                limit, ok = "Memory capacity", False
            elif tps > 0:
                w_read = weights * (1 - (1 - frac) ** max(1, b))
                if rp * g["bw"] * C["bwEff"] / (w_read + kv_user * b) < tps:
                    limit, ok = "Memory bandwidth (token speed)", False
                elif N * tflops * 1e12 * C["mfuDecode"] / (2 * m["active"] * 1e9) < users * tps:
                    limit, ok = "Compute (TFLOPS)", False
            if ok:
                break
            R = R + 1 if R < 64 else min(math.ceil(R * 1.02), max(R + 1, users))
        R = min(R, users)
        return dict(rep=rp, R=R, gpus=R * rp,
                    limit="Model weights (minimum footprint)" if (R == 1 and limit == "Memory capacity") else limit)

    best = solve(rep)
    rp = rep
    for _ in range(4):
        rp = rp * 2 if rp < inp["gps"] else rp + max(inp["gps"], 1)
        if rp > inp["gps"] and rep <= inp["gps"]:
            break
        s = solve(rp)
        if s["gpus"] < best["gpus"]:
            best = s

    res.update(rep=best["rep"], replicas=best["R"], baseGpus=best["gpus"], limit=best["limit"])
    b = users / res["replicas"]
    w_read = weights * (1 - (1 - frac) ** max(1, b))
    bw_tps = res["rep"] * g["bw"] * C["bwEff"] / (w_read + kv_user * b)
    comp_user = (res["rep"] * tflops * 1e12 * C["mfuDecode"] / (2 * m["active"] * 1e9)) / max(1, b)
    res["perUserTps"] = min(bw_tps, comp_user)
    res["aggTps"] = res["perUserTps"] * users
    res["ttft"] = (2 * m["active"] * 1e9 * inp["isl"]) / (res["rep"] * tflops * 1e12 * C["mfuPrefill"]) if inp["isl"] > 0 else 0
    res["memUsedGB"] = res["replicas"] * weights * ovh + kv_total * ovh

    per = max(res["rep"], inp["gps"] or 1)
    rr = lambda n: math.ceil(n / res["rep"]) * res["rep"]
    spare = {"none": 0, "n1": rr(per), "n2": rr(2 * per), "p10": rr(max(1, res["baseGpus"] * 0.10)),
             "p20": rr(max(1, res["baseGpus"] * 0.20)), "2n": res["baseGpus"]}.get(inp.get("redund") or "none", 0)
    res["spareGpus"] = spare
    res["gpus"] = res["baseGpus"] + spare
    res["servers"] = math.ceil(res["gpus"] / inp["gps"]) if inp["gps"] > 0 else math.nan
    res["memAvailGB"] = res["gpus"] * usable
    res["gpuKW"] = res["gpus"] * g["tdp"] / 1000
    res["itKW"] = res["gpuKW"] * (1 + C["hostOverhead"])
    res["facKW"] = res["itKW"] * C["pue"]
    res["fitsServer"] = res["rep"] <= inp["gps"]
    _host_and_totals(res, m, inp, data)
    return res


def _host_and_totals(res: dict, m: dict, inp: dict, data: dict) -> None:
    C, g = data["constants"], res["gpu"]
    raw = g.get("memRaw") or g["mem"]
    per_srv = min(inp["gps"] or 1, res["gpus"])
    res["gpuMemPerSrv"] = per_srv * raw
    res["totalGpuMemRaw"] = res["gpus"] * raw
    res["totalGpuMemUsable"] = res["gpus"] * res["usable"]
    res["totalBW_TBs"] = res["gpus"] * g["bw"] / 1000
    res["totalTflops"] = res["gpus"] * res["tflops"]
    res["totalTflops16"] = res["gpus"] * g["fp16"]
    res["totalTflops8"] = res["gpus"] * g["fp8"]
    res["totalTflops4"] = res["gpus"] * (g.get("fp4") or 0)
    res["totalTflops32"] = res["gpus"] * g["fp32"]
    tps = inp.get("tps") or 0
    res["reqTflops"] = res["users"] * tps * 2 * m["active"] * 1e9 / 1e12 if tps > 0 else 0
    res["computeUtil"] = res["reqTflops"] / res["totalTflops"] if res["reqTflops"] > 0 else 0
    if g["form"] == "NVL72":
        res["hostMinGB"] = res["hostRecGB"] = 240 * per_srv
        res["hostIntegrated"] = True
    else:
        cap = C["hostRamCapPer8"] * math.ceil(per_srv / 8)
        res["hostMinGB"] = max(128, _ram_step(res["gpuMemPerSrv"] * C["hostRamMin"], data))
        res["hostRecGB"] = max(256, res["hostMinGB"], min(cap, _ram_step(res["gpuMemPerSrv"] * C["hostRamRec"], data)))
    res["hostTotalRecGB"] = res["hostRecGB"] * res["servers"]
    res["cpuCores"] = max(16, per_srv * C["coresPerGPU"])


def cost_estimate(res: dict, inp: dict, data: dict = None) -> dict | None:
    """ROUGH indicative cost ranges (see COST_CAVEAT). Mirrors ``costEst()`` in js/app.js."""
    if res.get("infeasible"):
        return None
    data = data or DATA
    C, g, ci = data["constants"], res["gpu"], inp.get("cost") or {}
    if ci.get("gpu") == g["id"] and (ci.get("plo") or 0) > 0 and (ci.get("phi") or 0) > 0:
        plo, phi = ci["plo"], ci["phi"]
    else:
        plo, phi = g["price"]
    base = data["serverBaseCostPer8"].get(g["form"], data["serverBaseCostPer8"]["SXM"])
    slots = res["servers"] * (res["gps"] or 1)
    years = ci.get("years") or 3
    kwh = 0.12 if ci.get("kwh") is None else ci["kwh"]
    util = (ci.get("util") or 50) / 100
    maint = (10 if ci.get("maint") is None else ci["maint"]) / 100
    hw_lo = res["gpus"] * plo + slots / 8 * base[0]
    hw_hi = res["gpus"] * phi + slots / 8 * base[1]
    cap_lo, cap_hi = hw_lo * (1 + C["netLo"]), hw_hi * (1 + C["netHi"])
    hours = 8760 * years
    en_lo, en_hi = res["facKW"] * C["drawLo"] * hours * kwh, res["facKW"] * C["drawHi"] * hours * kwh
    m_lo, m_hi = cap_lo * maint * 0.8 * years, cap_hi * maint * 1.2 * years
    tco_lo, tco_hi = cap_lo + en_lo + m_lo, cap_hi + en_hi + m_hi
    tokens = res["aggTps"] * util * hours * 3600
    return dict(plo=plo, phi=phi, gpuLo=res["gpus"] * plo, gpuHi=res["gpus"] * phi, capLo=cap_lo, capHi=cap_hi,
                enLo=en_lo, enHi=en_hi, mLo=m_lo, mHi=m_hi, tcoLo=tco_lo, tcoHi=tco_hi, mid=(tco_lo + tco_hi) / 2,
                perMLo=tco_lo / tokens * 1e6 if tokens > 0 else math.nan,
                perMHi=tco_hi / tokens * 1e6 if tokens > 0 else math.nan, years=years, kwh=kwh, util=util, maint=maint)


def compute_all(inp: dict, data: dict = None) -> dict:
    """Size the workload on every GPU, rank them and pick the recommendation (``computeAll()``)."""
    data = data or DATA
    m = model_from(inp, data)
    rows = []
    for g in data["gpus"]:
        r = size(m, g, inp, data)
        r["cost"] = cost_estimate(r, inp, data)
        rows.append(r)
    rank_by = inp.get("rankBy") or "servers"
    cmid = lambda r: r["cost"]["mid"] if r["cost"] else INF
    keys = {
        "servers": lambda r: (r["infeasible"], not r["fitsServer"], r["servers"], r["facKW"], r["gpus"]),
        "power": lambda r: (r["infeasible"], r["facKW"], r["servers"]),
        "cost": lambda r: (r["infeasible"], cmid(r), r["facKW"]),
        "gpus": lambda r: (r["infeasible"], r["gpus"], r["facKW"]),
    }
    ranked = sorted(rows, key=keys[rank_by])
    sel = next(r for r in rows if r["gpu"]["id"] == inp["gpu"])
    return dict(inp=inp, model=m, all=rows, ranked=ranked, best=ranked[0], sel=sel)


def sweep(inp: dict, param: str, values: Iterable, gpus: Iterable[str] | None = None, data: dict = None):
    """Vary one input (e.g. ``users``) and size each GPU. Returns a pandas DataFrame."""
    import pandas as pd
    data = data or DATA
    out = []
    gids = list(gpus) if gpus else [g["id"] for g in data["gpus"]]
    for v in values:
        i = deepcopy(inp)
        i[param] = v
        m = model_from(i, data)
        for gid in gids:
            r = size(m, _gpu(gid, data), i, data)
            out.append({param: v, "gpu": r["gpu"]["name"], "gpus": None if r["infeasible"] else r["gpus"],
                        "servers": None if r["infeasible"] else r["servers"],
                        "facility_kW": None if r["infeasible"] else r["facKW"], "limited_by": r["limit"]})
    return pd.DataFrame(out)


def plan(scenarios: list[dict], data: dict = None) -> dict:
    """Multi-workload plan: each scenario ({'name', 'inp'}) sized on its own selected GPU, plus totals."""
    import pandas as pd
    data = data or DATA
    res = [(s["name"], compute_all(s["inp"], data)) for s in scenarios]
    rows = []
    for name, R in res:
        r = R["sel"]
        rows.append({"workload": name, "model": R["model"]["name"], "precision": R["inp"]["prec"], "users": R["inp"]["users"],
                     "gpu": r["gpu"]["name"], "gpus": None if r["infeasible"] else r["gpus"],
                     "servers": None if r["infeasible"] else r["servers"],
                     "gpu_memory_GB": None if r["infeasible"] else r["totalGpuMemRaw"],
                     "host_RAM_total_GB": None if r["infeasible"] else r["hostTotalRecGB"],
                     "facility_kW": None if r["infeasible"] else round(r["facKW"], 2),
                     "hw_cost_low_$": round(r["cost"]["capLo"]) if r["cost"] else None,
                     "hw_cost_high_$": round(r["cost"]["capHi"]) if r["cost"] else None,
                     "limited_by": r["limit"]})
    df = pd.DataFrame(rows)
    alt = []
    for g in data["gpus"]:
        tot = dict(gpu=g["name"], gpus=0, servers=0, facility_kW=0.0, feasible=True)
        for _, R in res:
            r = next(a for a in R["all"] if a["gpu"]["id"] == g["id"])
            if r["infeasible"]:
                tot["feasible"] = False
            else:
                tot["gpus"] += r["gpus"]; tot["servers"] += r["servers"]; tot["facility_kW"] += r["facKW"]
        alt.append(tot)
    alt_df = pd.DataFrame(alt).sort_values(["feasible", "servers", "facility_kW"], ascending=[False, True, True]).reset_index(drop=True)
    return dict(workloads=df, totals=df[["gpus", "servers", "gpu_memory_GB", "host_RAM_total_GB", "facility_kW",
                                         "hw_cost_low_$", "hw_cost_high_$"]].sum(numeric_only=True),
                single_gpu_options=alt_df, results=res)


# --------------------------------------------------------------------------- tables
def comparison_table(result: dict):
    """All GPUs for one result, ranked, as a pandas DataFrame (same columns as the web app's CSV)."""
    import pandas as pd
    out = []
    for i, r in enumerate(result["ranked"], 1):
        c, bad = r["cost"], r["infeasible"]
        out.append({"rank": i, "gpu": r["gpu"]["name"], "vendor": r["gpu"]["vendor"], "form": r["gpu"]["form"],
                    "memory_GB": r["gpu"].get("memRaw") or r["gpu"]["mem"],
                    "gpus_per_copy": None if bad else r["rep"], "total_gpus": None if bad else r["gpus"],
                    "spare_gpus": r.get("spareGpus", 0), "servers": None if bad else r["servers"],
                    "total_gpu_mem_GB": None if bad else r["totalGpuMemRaw"],
                    "total_TFLOPS": None if bad else round(r["totalTflops"]),
                    "host_RAM_per_server_GB": None if bad else r["hostRecGB"],
                    "facility_kW": None if bad else round(r["facKW"], 2),
                    "tok_s_per_user": None if bad else round(r["perUserTps"], 1),
                    "hw_cost_low_$": round(c["capLo"]) if c else None, "hw_cost_high_$": round(c["capHi"]) if c else None,
                    "tco_low_$": round(c["tcoLo"]) if c else None, "tco_high_$": round(c["tcoHi"]) if c else None,
                    "limited_by": r["limit"]})
    return pd.DataFrame(out)


def models_table(data: dict = None):
    import pandas as pd
    return pd.DataFrame([{k: m.get(k) for k in ("id", "name", "total", "active", "kv", "state", "open", "est")}
                         for m in (data or DATA)["models"]])


def gpus_table(data: dict = None):
    import pandas as pd
    return pd.DataFrame([{k: g.get(k) for k in ("id", "name", "vendor", "arch", "form", "mem", "memRaw", "bw", "fp16",
                                                "fp8", "fp4", "fp32", "tdp", "price")} for g in (data or DATA)["gpus"]])


if __name__ == "__main__":  # tiny CLI: python pangolin_planner.py llama31-8b h100 500
    import sys
    a = sys.argv[1:]
    inp = default_inputs(**({"model": a[0]} if a else {}), **({"gpu": a[1]} if len(a) > 1 else {}),
                         **({"users": int(a[2])} if len(a) > 2 else {}))
    R = compute_all(inp)
    s, b = R["sel"], R["best"]
    print(f"Pangolin GPU Planner {__version__} – {R['model']['name']}, {inp['users']} users, {inp['prec']}")
    print(f"Selected {s['gpu']['name']}: {s['gpus']} GPUs, {s['servers']} servers, {s['facKW']:.1f} kW, limited by {s['limit']}")
    print(f"Recommended: {b['gpu']['name']} x {b['gpus']}")
