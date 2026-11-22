"""Regression tests for the Python sizing engine.

Reference values were produced by the web app (js/app.js) and the Python port, which were
verified to give identical results on 60 random scenarios across all 22 GPUs. If you change
the engine or js/data.js on purpose, update these values and note the change in CHANGELOG.md.
"""
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))
import pangolin_planner as pp  # noqa: E402

CASES = {
    # name: (inputs, best_gpu_id, best_gpus, selected_gpus, selected_servers, selected_kW, limit)
    "llama_h100_default": ({}, "a30", 2, 1, 1, 1.183, "Model weights (minimum footprint)"),
    "kimi_fp4_b300_n1": (dict(model="kimi-k31", prec="FP4", users=1000, gpu="b300", gps=8, tps=20, redund="n1", peak=1.5),
                         "mi355x", 16, 24, 3, 56.784, "Memory capacity"),
    "qwen_mi355x_p20": (dict(model="qwen38-27b", prec="FP8", users=5000, gpu="mi355x", gps=8, tps=30, redund="p20"),
                        "b100", 8, 6, 1, 14.196, "Memory bandwidth (token speed)"),
    "deepseek_rubin_2n": (dict(model="deepseek-v41", prec="FP8", users=10000, gpu="rubin", gps=72, tps=50, redund="2n"),
                          "rubin", 8, 8, 1, 24.336, "Model weights (minimum footprint)"),
    "llama_l4_fast": (dict(model="llama31-8b", users=100, gpu="l4", gps=2, tps=100),
                      "b100", 1, 32, 16, 3.894, "Memory bandwidth (token speed)"),
}


@pytest.mark.parametrize("name", CASES)
def test_reference_cases(name):
    inp, best_id, best_gpus, gpus, servers, kw, limit = CASES[name]
    R = pp.compute_all(pp.default_inputs(**inp))
    assert R["best"]["gpu"]["id"] == best_id
    assert R["best"]["gpus"] == best_gpus
    s = R["sel"]
    assert (s["gpus"], s["servers"]) == (gpus, servers)
    assert s["facKW"] == pytest.approx(kw, abs=1e-3)
    assert s["limit"] == limit


def test_data_file_integrity():
    d = pp.DATA
    assert len(d["models"]) == 25 and len(d["gpus"]) == 22
    ids = [g["id"] for g in d["gpus"]] + [m["id"] for m in d["models"]]
    assert len(ids) == len(set(ids)), "duplicate ids"
    for g in d["gpus"]:
        assert g["mem"] > 0 and g["bw"] > 0 and g["fp16"] > 0 and g["tdp"] > 0
        assert g["form"] in d["maxGpusPerServer"]
        assert len(g["price"]) == 2 and g["price"][0] <= g["price"][1]
    for m in d["models"]:
        assert 0 < m["active"] <= m["total"]


def test_infeasible_speed_target():
    # A 70B dense model on L4 tops out near 96 tok/s per user even with a 64-GPU copy
    inp = pp.default_inputs(model="llama2-13b", ov={"total": 70, "active": 70}, gpu="l4", tps=100, gps=8)
    r = pp.compute_all(inp)["sel"]
    assert r["infeasible"] and math.isinf(r["gpus"])


def test_more_users_never_needs_fewer_gpus():
    prev = 0
    for u in (10, 100, 1000, 10000):
        g = pp.compute_all(pp.default_inputs(model="glm53", prec="FP8", users=u, gpu="b200", gps=8))["sel"]["gpus"]
        assert g >= prev
        prev = g


def test_redundancy_adds_spares():
    base = pp.compute_all(pp.default_inputs(gps=8, users=1000))["sel"]
    n1 = pp.compute_all(pp.default_inputs(gps=8, users=1000, redund="n1"))["sel"]
    two_n = pp.compute_all(pp.default_inputs(gps=8, users=1000, redund="2n"))["sel"]
    assert n1["gpus"] > base["gpus"] and two_n["gpus"] == 2 * base["gpus"]


def test_cost_is_a_range():
    c = pp.compute_all(pp.default_inputs())["sel"]["cost"]
    assert 0 < c["capLo"] < c["capHi"] and c["tcoLo"] < c["tcoHi"]


def test_scenario_roundtrip(tmp_path):
    import json
    f = tmp_path / "s.json"
    f.write_text(json.dumps({"version": 2, "scenarios": [{"name": "x", "inp": pp.default_inputs(users=500)}]}))
    sc = pp.load_scenarios(f)
    P = pp.plan(sc)
    assert len(P["workloads"]) == 1
