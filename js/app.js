/* Pangolin GPU Planner – application logic (sizing engine, UI, exports).
 * The Python port in python/pangolin_planner.py mirrors size()/costEst()/computeAll(). Keep them in step. */
/* Data & assumptions live in js/data.js (shared with the Python engine). */
const PD = window.PLANNER_DATA;
const DATA_DATE = PD.dataDate, MODELS = PD.models, GPUS = PD.gpus;
const MAX_PER_SERVER = PD.maxGpusPerServer, SERVER_BASE = PD.serverBaseCostPer8;
const C = PD.constants, BPP = PD.bytesPerParam, KVB = PD.kvBytes, PREC_LABEL = PD.precisionLabels, RAM_STEPS = PD.ramSteps;
const ramStep = gb => RAM_STEPS.find(x=>x>=gb) || Math.ceil(gb/4096)*4096;

/* ---------------- helpers ---------------- */
const $ = id => document.getElementById(id);
const fmt = (n,d=0) => (isFinite(n)? Number(n).toLocaleString(undefined,{maximumFractionDigits:d,minimumFractionDigits:d}) : "–");
const gbtb = gb => !isFinite(gb)? "–" : gb>=1000 ? fmt(gb/1000, gb>=10000?0:1)+" TB" : fmt(gb, gb<10?1:0)+" GB";
const ramf = gb => !isFinite(gb)? "–" : gb>=1024 ? fmt(gb/1024, (gb%1024)?1:0)+" TB" : fmt(gb)+" GB";
const tf = t => !isFinite(t)? "–" : t>=1000 ? fmt(t/1000, t>=100000?0:1)+" PFLOPS" : fmt(t,0)+" TFLOPS";
const money = v => !isFinite(v)? "–" : v>=1e9 ? "$"+fmt(v/1e9,2)+"B" : v>=1e6 ? "$"+fmt(v/1e6,v>=1e8?0:1)+"M" : v>=1e3 ? "$"+fmt(v/1e3,0)+"K" : "$"+fmt(v,v<10?2:0);
const range = (a,b) => (!isFinite(a)||!isFinite(b)) ? "–" : `${money(a)} – ${money(b)}`;
const esc = s => String(s??"").replace(/[&<>"]/g, c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
function fill(sel, values, def, label=v=>fmt(v)){
  sel.innerHTML = values.map(v=>`<option value="${v}" ${v==def?"selected":""}>${label(v)}${v==def?" (default)":""}</option>`).join("");
}
function toast(t){ const el=$("toast"); el.textContent=t; el.style.display="block"; clearTimeout(el._t); el._t=setTimeout(()=>el.style.display="none",2800); }
function download(name, blob){
  const a=document.createElement("a"); a.href=URL.createObjectURL(blob); a.download=name; document.body.appendChild(a); a.click();
  setTimeout(()=>{URL.revokeObjectURL(a.href); a.remove();},1000);
}
const store = { get(k,d){ try{ const v=localStorage.getItem(k); return v?JSON.parse(v):d; }catch(e){ return d; } },
                set(k,v){ try{ localStorage.setItem(k,JSON.stringify(v)); }catch(e){} } };

/* ---------------- UI setup ---------------- */
$("model").innerHTML = MODELS.map(m=>`<option value="${m.id}">${m.name}${m.open?"":" · closed"}${m.est?" · est.":""}</option>`).join("");
const vendors=[...new Set(GPUS.map(g=>g.vendor))];
$("gpu").innerHTML = vendors.map(v=>`<optgroup label="${v}">`+GPUS.filter(g=>g.vendor===v).map(g=>`<option value="${g.id}">${g.name} – ${g.memRaw||g.mem} GB${g.prelim?" (preliminary)":""}</option>`).join("")+"</optgroup>").join("");
const SEQ=[0,128,256,512,1024,2048,3072,4096,6144,8192];
const USERS=[1,5,10,25,50,100,250,500,1000,2500,5000,10000,25000,50000,100000];
const UTIL=[50,60,70,75,80,85,90,95,100];
const GPS=[1,2,4,8,16,32,64,72,128,256,512,1024];

function setDefaults(){
  $("model").value="llama31-8b"; $("precision").value="FP16"; $("usecase").value="custom";
  fill($("isl"),SEQ,1024); fill($("osl"),SEQ,1024); fill($("users"),USERS,100);
  fill($("util"),UTIL,100,v=>v+"%"); fill($("gps"),GPS,2); $("gpu").value="h100"; $("tps").value="20";
  $("peak").value="1"; $("redund").value="none"; $("rankBy").value="servers";
  $("cKwh").value="0.12"; $("cYears").value="3"; $("cUtil").value="50"; $("cMaint").value="10";
  onModel(); onGpu();
  $("report").style.display="none";
}
function onModel(){
  const m = MODELS.find(x=>x.id===$("model").value);
  $("modelHint").textContent = `${fmt(m.total,m.total<100?1:0)}B total / ${fmt(m.active,m.active<100?1:0)}B active · ${m.open?"open weights":"closed (API only)"}`;
  $("ovTotal").value=m.total; $("ovActive").value=m.active; $("ovKv").value=m.kv; $("ovState").value=m.state||0;
  $("ovMsg").innerHTML = m.est
      ? "⚠ This model's size has not been published. The figures below are <b>placeholder estimates</b>. Replace them if you have real figures."
      : (m.stateEst||m.kvEst) ? "Published parameter counts. <b>Some cache/state figures are estimated</b>. Adjust them if you have the model's config."
      : "Published model figures. Edit them only to model a variant (e.g. a fine-tune or a different checkpoint).";
}
function onGpu(){
  const g=GPUS.find(x=>x.id===$("gpu").value);
  $("gpuHint").textContent=`${g.arch} · ${g.form} · ${g.memRaw||g.mem} GB · ${fmt(g.bw/1000,2)} TB/s · ${g.tdp} W${g.fp4?" · native FP4":""}`;
  $("cPlo").value=g.price[0]; $("cPhi").value=g.price[1];
  $("gpsHint").textContent = g.form==="NVL72" ? "Rack-scale GPU: use 72 to model whole NVL72 racks." : "Range 0–1024 · default 2";
  onPrec();
}
function onPrec(){
  const p=$("precision").value, g=GPUS.find(x=>x.id===$("gpu").value);
  $("precHint").textContent = p==="FP32" ? "FP32 is rarely used for inference: it needs 2× the memory of FP16."
    : p==="FP8" ? (g.fp8native ? "Native FP8 on the selected GPU." : `${g.arch} has no native FP8, so this uses INT8 quantisation.`)
    : p==="FP4" ? (g.fp4 ? "Native FP4 tensor cores on the selected GPU." : `${g.name} has no native FP4: weights are stored in 4-bit (weight-only INT4) and computed at ${g.fp8native?"FP8":"FP16"} rate.`)
    : "Standard production precision.";
}
$("model").onchange=onModel; $("gpu").onchange=onGpu; $("precision").onchange=onPrec;
$("usecase").onchange=()=>{ const o=$("usecase").selectedOptions[0]; if(o.dataset.in){ $("isl").value=o.dataset.in; $("osl").value=o.dataset.out; } };
["isl","osl"].forEach(id=>$(id).addEventListener("change",()=>{$("usecase").value="custom";}));

/* ---------------- read / apply inputs (also used for scenarios) ---------------- */
function readInputs(){
  return {
    model:$("model").value, ov:{total:+$("ovTotal").value, active:+$("ovActive").value, kv:+$("ovKv").value, state:+$("ovState").value},
    prec:$("precision").value, usecase:$("usecase").value, isl:+$("isl").value, osl:+$("osl").value, users:+$("users").value,
    tps:+$("tps").value, peak:+$("peak").value, gpu:$("gpu").value, util:+$("util").value, gps:+$("gps").value,
    redund:$("redund").value, rankBy:$("rankBy").value,
    cost:{kwh:+$("cKwh").value, years:+$("cYears").value, util:+$("cUtil").value, maint:+$("cMaint").value, plo:+$("cPlo").value, phi:+$("cPhi").value, gpu:$("gpu").value}
  };
}
function ensureOpt(sel,val){ if(![...sel.options].some(o=>o.value==val)){ const o=document.createElement("option"); o.value=val; o.textContent=fmt(+val)||val; sel.appendChild(o);} sel.value=val; }
function applyInputs(i){
  $("model").value=i.model; onModel();
  if(i.ov){ $("ovTotal").value=i.ov.total; $("ovActive").value=i.ov.active; $("ovKv").value=i.ov.kv; $("ovState").value=i.ov.state; }
  $("precision").value=i.prec; $("usecase").value=i.usecase||"custom";
  ensureOpt($("isl"),i.isl); ensureOpt($("osl"),i.osl); ensureOpt($("users"),i.users); ensureOpt($("util"),i.util); ensureOpt($("gps"),i.gps);
  $("tps").value=i.tps; $("peak").value=i.peak||1; $("gpu").value=i.gpu; onGpu(); $("redund").value=i.redund||"none"; $("rankBy").value=i.rankBy||"servers";
  if(i.cost){ $("cKwh").value=i.cost.kwh; $("cYears").value=i.cost.years; $("cUtil").value=i.cost.util; $("cMaint").value=i.cost.maint;
              if(i.cost.gpu===i.gpu){ $("cPlo").value=i.cost.plo; $("cPhi").value=i.cost.phi; } }
  onPrec();
}
function modelFrom(i){
  const base = MODELS.find(x=>x.id===i.model); const m={...base};
  const o=i.ov||{};
  if(o.total>0) m.total=o.total; if(o.active>0) m.active=Math.min(o.active,m.total); if(o.kv>=0) m.kv=o.kv; if(o.state>=0) m.state=o.state;
  m.edited = ["total","active","kv"].some(k=>o[k]>0 && Math.abs(o[k]-base[k])>1e-9) || (o.state>=0 && Math.abs(o.state-(base.state||0))>1e-9);
  return m;
}

/* ---------------- sizing engine ---------------- */
function nextPow2(n){ let p=1; while(p<n) p*=2; return p; }
function tflopsFor(g,p){ return p==="FP32"? g.fp32 : p==="FP8"? g.fp8 : p==="FP4"? (g.fp4 || (g.fp8native? g.fp8 : g.fp16)) : g.fp16; }
function size(m, g, inp0){
  const maxPS = MAX_PER_SERVER[g.form]||8;
  const inp = {...inp0, gps: Math.max(1, Math.min(inp0.gps||1, maxPS))};
  const users = Math.ceil(inp.users*(inp.peak||1));
  const bpp = BPP[inp.prec], tflops = tflopsFor(g,inp.prec);
  const usable = g.mem * inp.util/100;
  const weightsGB = m.total*bpp, activeGB = m.active*bpp;
  const kvTokB = m.kv*1024*(KVB[inp.prec]/2);
  const ctx = inp.isl + inp.osl;
  const stateGB = (m.state||0)/1000;
  const kvUserGB = kvTokB*ctx/1e9 + stateGB;
  const kvTotalGB = kvUserGB*users;
  const res = {g, gps:inp.gps, gpsCapped: inp.gps < (inp0.gps||1), users, tflops, usable, weightsGB, activeGB, kvTokB, stateGB, kvUserGB, kvTotalGB, ctx};
  if(usable<=0){ return markInfeasible(res,"Memory utilisation is 0%"); }

  const minMem = weightsGB*(1+C.overhead) + kvUserGB*(1+C.overhead);
  let rep = nextPow2(Math.ceil(minMem/usable));
  const speedAt1 = r => Math.min(r*g.bw*C.bwEff/(activeGB + kvUserGB), r*tflops*1e12*C.mfuDecode/(2*m.active*1e9));
  if(inp.tps>0){ while(speedAt1(rep) < inp.tps && rep < 64) rep*=2; }
  if(rep > inp.gps && inp.gps>0) rep = Math.ceil(rep/inp.gps)*inp.gps;
  res.rep = rep;
  if(inp.tps>0 && speedAt1(rep) < inp.tps){ res.perUserTps = speedAt1(rep); return markInfeasible(res,`Cannot reach ${inp.tps} tok/s per user`); }

  const frac = m.active/m.total;
  const memNeedTotal = kvTotalGB*(1+C.overhead);
  const compNeed = inp.tps>0 ? users*inp.tps*2*m.active*1e9/(tflops*1e12*C.mfuDecode) : 0;
  function solve(rp){
    let R = Math.max(1, Math.floor(Math.min(memNeedTotal/(rp*usable), compNeed/rp || Infinity)*0.5)), limit="Memory capacity";
    if(!isFinite(R)) R=1;
    for(; R<=users; R = R<64 ? R+1 : Math.min(Math.ceil(R*1.02), Math.max(R+1, users))){
      const N=R*rp, b=users/R;
      if(rp*usable < weightsGB*(1+C.overhead) + kvUserGB*b*(1+C.overhead)){ limit="Memory capacity"; continue; }
      if(inp.tps>0){
        const wRead = weightsGB*(1-Math.pow(1-frac, Math.max(1,b)));   // unique (expert) weights touched per decode step
        if(rp*g.bw*C.bwEff/(wRead + kvUserGB*b) < inp.tps){ limit="Memory bandwidth (token speed)"; continue; }
        if(N*tflops*1e12*C.mfuDecode/(2*m.active*1e9) < users*inp.tps){ limit="Compute (TFLOPS)"; continue; }
      }
      break;
    }
    R=Math.min(R,users);
    return {rep:rp, R, gpus:R*rp, limit:(R===1 && limit==="Memory capacity")?"Model weights (minimum footprint)":limit};
  }
  let best = solve(rep);
  for(let k=0, rp=rep; k<4; k++){
    rp = rp < inp.gps ? rp*2 : rp + Math.max(inp.gps,1);
    if(rp > inp.gps && rep <= inp.gps) break;
    const s=solve(rp); if(s.gpus < best.gpus) best=s;
  }
  res.rep=best.rep; res.replicas=best.R; res.baseGpus=best.gpus; res.limit=best.limit;
  // performance at the sized (base) configuration
  const b = users/res.replicas;
  const wRead = weightsGB*(1-Math.pow(1-frac,Math.max(1,b)));
  const bwTps = res.rep*g.bw*C.bwEff/(wRead + kvUserGB*b);
  const compUser = (res.rep*tflops*1e12*C.mfuDecode/(2*m.active*1e9))/Math.max(1,b);
  res.perUserTps = Math.min(bwTps, compUser);
  res.aggTps = res.perUserTps*users;
  res.ttft = inp.isl>0 ? (2*m.active*1e9*inp.isl)/(res.rep*tflops*1e12*C.mfuPrefill) : 0;
  res.memUsedGB = res.replicas*weightsGB*(1+C.overhead) + kvTotalGB*(1+C.overhead);
  // redundancy
  const per = Math.max(res.rep, inp.gps||1);
  const roundRep = n => Math.ceil(n/res.rep)*res.rep;
  const spare = {none:0, n1:roundRep(per), n2:roundRep(2*per), p10:roundRep(Math.max(1,res.baseGpus*0.10)), p20:roundRep(Math.max(1,res.baseGpus*0.20)), "2n":res.baseGpus}[inp.redund||"none"]||0;
  res.spareGpus = spare; res.gpus = res.baseGpus + spare;
  res.servers = inp.gps>0 ? Math.ceil(res.gpus/inp.gps) : NaN;
  res.memAvailGB = res.gpus*usable;
  res.gpuKW = res.gpus*g.tdp/1000; res.itKW = res.gpuKW*(1+C.hostOverhead); res.facKW = res.itKW*C.pue;
  res.fitsServer = res.rep <= inp.gps;
  hostAndTotals(res, m, inp);
  return res;
}
function markInfeasible(res, why){
  Object.assign(res,{infeasible:true, limit:why, gpus:Infinity, baseGpus:Infinity, spareGpus:0, replicas:NaN, servers:Infinity, facKW:Infinity, gpuKW:Infinity, itKW:Infinity,
    aggTps:NaN, ttft:NaN, memUsedGB:NaN, memAvailGB:NaN, fitsServer:false, totalGpuMemRaw:NaN, totalTflops:NaN, hostRecGB:NaN, hostMinGB:NaN});
  if(!isFinite(res.perUserTps)) res.perUserTps=NaN;
  return res;
}
function hostAndTotals(res, m, inp){
  const g=res.g, raw=g.memRaw||g.mem;
  const gpusPerSrv = Math.min(inp.gps||1, res.gpus);
  res.gpuMemPerSrv = gpusPerSrv*raw;
  res.totalGpuMemRaw = res.gpus*raw; res.totalGpuMemUsable = res.gpus*res.usable;
  res.totalBW = res.gpus*g.bw/1000;
  res.totalTflops = res.gpus*res.tflops;
  res.totalTflops16 = res.gpus*g.fp16; res.totalTflops8 = res.gpus*g.fp8; res.totalTflops4 = res.gpus*(g.fp4||0); res.totalTflops32 = res.gpus*g.fp32;
  res.reqTflops = inp.tps>0 ? res.users*inp.tps*2*m.active*1e9/1e12 : 0;
  res.computeUtil = res.reqTflops>0 ? res.reqTflops/res.totalTflops : 0;
  if(g.form==="NVL72"){   // Grace/Vera CPUs: LPDDR5X per superchip (approx. 480 GB per 2 GPUs)
    res.hostMinGB = res.hostRecGB = 240*gpusPerSrv; res.hostIntegrated=true;
  } else {
    const cap = C.hostRamCapPer8*Math.ceil(gpusPerSrv/8);
    res.hostMinGB = Math.max(128, ramStep(res.gpuMemPerSrv*C.hostRamMin));
    res.hostRecGB = Math.max(256, res.hostMinGB, Math.min(cap, ramStep(res.gpuMemPerSrv*C.hostRamRec)));
  }
  res.hostTotalRecGB = res.hostRecGB*res.servers;
  res.cpuCores = Math.max(16, gpusPerSrv*C.coresPerGPU);
}

/* ---------------- rough cost estimate ---------------- */
function costEst(res, inp){
  if(res.infeasible) return null;
  const g=res.g, ci=inp.cost||{};
  const [plo,phi] = (ci.gpu===g.id && ci.plo>0 && ci.phi>0) ? [ci.plo,ci.phi] : g.price;
  const base = SERVER_BASE[g.form]||SERVER_BASE.SXM;
  const slots = res.servers*(res.gps||1);
  const years = ci.years||3, kwh = ci.kwh??0.12, util=(ci.util||50)/100, maint=(ci.maint??10)/100;
  const hwLo = res.gpus*plo + slots/8*base[0], hwHi = res.gpus*phi + slots/8*base[1];
  const capLo = hwLo*(1+C.netLo), capHi = hwHi*(1+C.netHi);
  const hours = 8760*years;
  const enLo = res.facKW*C.drawLo*hours*kwh, enHi = res.facKW*C.drawHi*hours*kwh;
  const mLo = capLo*maint*0.8*years, mHi = capHi*maint*1.2*years;
  const tcoLo = capLo+enLo+mLo, tcoHi = capHi+enHi+mHi;
  const tokens = res.aggTps*util*hours*3600;
  return { plo, phi, gpuLo:res.gpus*plo, gpuHi:res.gpus*phi, capLo, capHi, enLo, enHi, mLo, mHi, tcoLo, tcoHi, mid:(tcoLo+tcoHi)/2,
           perMLo: tokens>0? tcoLo/tokens*1e6 : NaN, perMHi: tokens>0? tcoHi/tokens*1e6 : NaN, years, kwh, util, maint };
}

/* ---------------- full computation for one input set ---------------- */
function computeAll(inp){
  const m = modelFrom(inp);
  const all = GPUS.map(g=>{ const r=size(m,g,inp); r.cost=costEst(r,inp); return r; });
  const key = {
    servers:(a,b)=>(b.fitsServer-a.fitsServer)||(a.servers-b.servers)||(a.facKW-b.facKW)||(a.gpus-b.gpus),
    power:(a,b)=>(a.facKW-b.facKW)||(a.servers-b.servers),
    cost:(a,b)=>((a.cost?a.cost.mid:Infinity)-(b.cost?b.cost.mid:Infinity))||(a.facKW-b.facKW),
    gpus:(a,b)=>(a.gpus-b.gpus)||(a.facKW-b.facKW)
  }[inp.rankBy||"servers"];
  const ranked = [...all].sort((a,b)=> (a.infeasible-b.infeasible) || key(a,b));
  const sel = all.find(r=>r.g.id===inp.gpu);
  return { inp, m, all, ranked, best:ranked[0], sel, when:new Date() };
}

/* ---------------- branding ---------------- */
function brand(){ return {company:$("bCompany").value.trim(), project:$("bProject").value.trim(), author:$("bAuthor").value.trim()}; }
function brandHtml(title){
  const b=brand();
  const line=[b.company,b.project].filter(Boolean).join(" · ");
  return `<div class="brand">${line?`<b>${esc(line)}</b> · `:""}${b.author?`Prepared by ${esc(b.author)} · `:""}${new Date().toLocaleString()}</div>`;
}
const CAVEAT_HTML = `<div class="caveat"><b class="t">⚠ CAVEAT EMPTOR: COST FIGURES ARE ROUGH ESTIMATES.</b><br>
GPU, server, network and energy prices change quickly and vary with supplier, volume, contract terms, region, configuration and availability. These ranges are for <b>early budgeting and relative comparison only</b>. They are not quotes and may be wrong by ±30–50% or more. Excluded: facility build-out, cooling plant, software licences, staffing, taxes, financing. Price data last reviewed ${DATA_DATE}.</div>`;

/* ---------------- single report ---------------- */
let LAST = null;
function generate(){
  const R = computeAll(readInputs()); LAST=R;
  const {m, inp, ranked, best, sel} = R;
  const precLabel = PREC_LABEL[inp.prec];
  const leanest = [...R.all].filter(r=>!r.infeasible).sort((a,b)=>a.facKW-b.facKW)[0];
  const cheapest = [...R.all].filter(r=>r.cost).sort((a,b)=>a.cost.mid-b.cost.mid)[0];
  const maxG = Math.max(1,...R.all.filter(r=>!r.infeasible).map(r=>r.gpus));
  const users = sel.users;
  const tpsLine = inp.tps>0 ? `${inp.tps} tokens/s per user` : "No speed target (memory-only)";
  const redundLabel = $("redund").querySelector(`option[value="${inp.redund}"]`).textContent;

  const W=[];
  if(!m.open) W.push(`<div class="badbox"><b>${m.name} is a closed, API-only model.</b> Its weights are not available to self-host, so this sizing is a what-if for a licensed on-premises deployment only. ${m.est?"The parameter figures are placeholder estimates, not vendor data.":""}</div>`);
  if(m.edited) W.push(`<div class="warnbox">Model figures have been edited from the built-in values (see Inputs &amp; model details).</div>`);
  if(sel.infeasible) W.push(`<div class="badbox"><b>${sel.g.name} cannot deliver ${inp.tps} tokens/s per user for this model</b> (max ≈${fmt(sel.perUserTps,1)} tok/s even with one model copy spread over ${sel.rep} GPUs). Choose a faster GPU, a lower speed target or a lower precision.</div>`);
  if(m.stateEst) W.push(`<div class="warnbox">${m.name}'s per-user recurrent state (${fmt(m.state)} MB) is an estimate and can dominate memory at high concurrency.</div>`);
  if(inp.prec==="FP32") W.push(`<div class="warnbox">FP32 doubles memory against FP16 and is almost never used in production inference.</div>`);
  if(inp.prec==="FP8" && !sel.g.fp8native) W.push(`<div class="warnbox">${sel.g.name} (${sel.g.arch}) has no native FP8. The sizing assumes INT8 quantisation.</div>`);
  if(inp.prec==="FP4") W.push(sel.g.fp4 ? `<div class="warnbox">FP4 assumes a quantised checkpoint (NVFP4/MXFP4) with FP8 KV cache. Validate accuracy for your use case, since 4-bit can reduce quality on some tasks.</div>`
                                     : `<div class="warnbox">${sel.g.name} has no native FP4. This assumes 4-bit weight-only quantisation (e.g. AWQ/GPTQ INT4), which saves memory but computes at ${sel.g.fp8native?"FP8":"FP16"} speed.</div>`);
  if(m.id==="kimi-k31" && inp.prec!=="FP4") W.push(`<div class="warnbox">Kimi K3.1-Pro ships natively as MXFP4 (~1.4 TB). FP4 precision on a Blackwell or MI355X GPU matches the official checkpoint.</div>`);
  if(!sel.infeasible && !sel.fitsServer) W.push(`<div class="warnbox">With ${sel.g.name}, one model copy needs <b>${sel.rep} GPUs</b>, more than the ${sel.gps} GPUs per server. The model must span servers over a high-speed fabric (InfiniBand / NVLink Switch). Consider more GPUs per server or a larger-memory GPU.</div>`);
  if(!sel.infeasible && sel.g.form==="PCIe" && sel.rep>1) W.push(`<div class="warnbox">${sel.g.name} is a PCIe card. Splitting one model copy across ${sel.rep} GPUs over PCIe is slower than modelled. Expect 10–30% lower throughput.</div>`);
  if(sel.gpsCapped) W.push(`<div class="warnbox">${sel.g.name} (${sel.g.form}) is limited to ${sel.gps} GPUs per server, so the ${inp.gps} entered was capped for this GPU. Other GPUs in the comparison are capped the same way (8 for PCIe/SXM/OAM servers, 72 for NVL72 racks).</div>`);
  if(inp.util===100) W.push(`<div class="warnbox">Memory utilisation is set to 100%. Serving engines usually reserve 5–10% headroom. Use 90% for a safer plan.</div>`);
  if(inp.redund==="none") W.push(`<div class="warnbox">No redundancy is included. A single server failure would reduce capacity. Consider N+1 for production.</div>`);
  if(sel.g.note) W.push(`<div class="warnbox">${sel.g.note}</div>`);

  const rows = ranked.map((r,i)=>`
    <tr class="${r.g.id===sel.g.id?"sel":""} ${i===0&&!r.infeasible?"top":""}">
      <td>${i+1}. ${r.g.name}${i===0&&!r.infeasible?'<span class="tag g">Recommended</span>':""}${r.g.id===sel.g.id?'<span class="tag">Selected</span>':""}${!r.infeasible&&!r.fitsServer?'<span class="tag w">multi-server copy</span>':""}${r.g.prelim?'<span class="tag w">prelim.</span>':""}</td>
      <td class="num">${r.g.memRaw||r.g.mem} GB</td>
      <td class="num">${r.infeasible?"–":fmt(r.rep)}</td>
      <td class="num"><b>${r.infeasible?"n/a":fmt(r.gpus)}</b>${r.spareGpus?`<div class="muted" style="font-size:11px">incl. ${fmt(r.spareGpus)} spare</div>`:""}</td>
      <td class="barcell">${r.infeasible?'<span class="tag b">not feasible</span>':`<div class="bar" style="width:${Math.max(1,r.gpus/maxG*100)}%"></div>`}</td>
      <td class="num">${fmt(r.servers)}${!r.infeasible&&r.gps!==inp.gps?`<div class="muted" style="font-size:11px">× ${r.gps}</div>`:""}</td>
      <td class="num">${r.infeasible?"–":gbtb(r.totalGpuMemRaw)}</td>
      <td class="num">${r.infeasible?"–":tf(r.totalTflops)}</td>
      <td class="num">${r.infeasible?"–":ramf(r.hostRecGB)}</td>
      <td class="num">${r.infeasible?"–":fmt(r.facKW,r.facKW<10?1:0)+" kW"}</td>
      <td class="num">${r.cost?range(r.cost.capLo,r.cost.capHi):"–"}</td>
      <td>${r.limit}</td>
    </tr>`).join("");

  const recBlock = best.infeasible
    ? `<div class="rec"><h3>No GPU in the list can meet ${inp.tps} tokens/s per user for this model</h3><div>Lower the speed target or precision and generate the report again.</div></div>`
    : `<div class="rec"><h3>Recommendation: ${best.g.name} × ${fmt(best.gpus)} GPUs</h3>
      <div>${fmt(best.servers)} server${best.servers===1?"":"s"} at ${best.gps} GPUs each · ${fmt(best.replicas)} model ${best.replicas===1?"copy":"copies"} of ${best.rep} GPU${best.rep>1?"s":""}${best.spareGpus?` + ${fmt(best.spareGpus)} spare GPUs`:""} · ≈${fmt(best.facKW,1)} kW facility · rough hardware cost ${range(best.cost.capLo,best.cost.capHi)}.
      ${best.g.id!==sel.g.id?(sel.infeasible?`<br>Your selected <b>${sel.g.name}</b> cannot meet the speed target.`:`<br>Your selected <b>${sel.g.name}</b> would need <b>${fmt(sel.gpus)} GPUs</b> across ${fmt(sel.servers)} servers (≈${fmt(sel.facKW,1)} kW).`):" This is also your selected GPU."}
      ${leanest && leanest.g.id!==best.g.id && leanest.facKW<best.facKW*0.98?`<br>Lowest-power option: <b>${leanest.g.name}</b>, ${fmt(leanest.gpus)} GPUs, ≈${fmt(leanest.facKW,1)} kW.`:""}
      ${cheapest && cheapest.g.id!==best.g.id && cheapest.cost.mid<best.cost.mid*0.95?`<br>Lowest rough-cost option: <b>${cheapest.g.name}</b>, ${fmt(cheapest.gpus)} GPUs, ${range(cheapest.cost.tcoLo,cheapest.cost.tcoHi)} ${cheapest.cost.years}-yr TCO.`:""}
      <div class="muted" style="font-size:12px;margin-top:4px">Ranking: ${$("rankBy").selectedOptions[0].textContent}.</div></div></div>`;

  const c = sel.cost;
  const costCard = c ? `
  <div class="card">
    <h2>Rough cost estimate: ${sel.g.name} deployment</h2>
    <p class="sub">Indicative ranges over ${c.years} years at ${fmt(c.util*100)}% average utilisation and $${fmt(c.kwh,2)}/kWh</p>
    ${CAVEAT_HTML}
    <div class="two">
      <table>
        <tr><th>Item</th><th class="num">Low</th><th class="num">High</th></tr>
        <tr><td>GPUs (${fmt(sel.gpus)} × ${money(c.plo)}–${money(c.phi)})</td><td class="num">${money(c.gpuLo)}</td><td class="num">${money(c.gpuHi)}</td></tr>
        <tr><td>Servers, network &amp; integration</td><td class="num">${money(c.capLo-c.gpuLo)}</td><td class="num">${money(c.capHi-c.gpuHi)}</td></tr>
        <tr><td><b>Hardware (CapEx)</b></td><td class="num"><b>${money(c.capLo)}</b></td><td class="num"><b>${money(c.capHi)}</b></td></tr>
        <tr><td>Electricity (${c.years} yr)</td><td class="num">${money(c.enLo)}</td><td class="num">${money(c.enHi)}</td></tr>
        <tr><td>Support &amp; maintenance (${c.years} yr)</td><td class="num">${money(c.mLo)}</td><td class="num">${money(c.mHi)}</td></tr>
        <tr class="total"><td>Total cost of ownership (${c.years} yr)</td><td class="num">${money(c.tcoLo)}</td><td class="num">${money(c.tcoHi)}</td></tr>
      </table>
      <table>
        <tr><th>Unit economics (very rough)</th><th class="num">Value</th></tr>
        <tr><td>Cost per 1M output tokens</td><td class="num">${money(c.perMLo)} – ${money(c.perMHi)}</td></tr>
        <tr><td>Cost per GPU-hour (TCO ÷ GPU-hours)</td><td class="num">${money(c.tcoLo/(sel.gpus*8760*c.years))} – ${money(c.tcoHi/(sel.gpus*8760*c.years))}</td></tr>
        <tr><td>Cost per concurrent user (TCO)</td><td class="num">${money(c.tcoLo/users)} – ${money(c.tcoHi/users)}</td></tr>
        <tr><td>Energy use per year</td><td class="num">${fmt(sel.facKW*C.drawLo*8760/1000,0)}–${fmt(sel.facKW*C.drawHi*8760/1000,0)} MWh</td></tr>
      </table>
    </div>
    <p class="muted" style="font-size:12px;margin-top:10px">Method: GPU price range (editable) + server base cost per GPU slot (${sel.g.form}: ${range((SERVER_BASE[sel.g.form]||[0,0])[0]/8,(SERVER_BASE[sel.g.form]||[0,0])[1]/8)} per slot) + ${C.netLo*100}–${C.netHi*100}% network/integration. Electricity at ${C.drawLo*100}–${C.drawHi*100}% of the facility power figure. Maintenance ±20% around the entered rate.</p>
  </div>` : "";

  const html = `
  <div class="card">
    <div class="rpt-head">
      <div>
        <h2>Pangolin GPU Planner report: ${m.name}</h2>
        <p class="sub" style="margin-bottom:4px">${precLabel} · ${fmt(inp.users)} concurrent users${inp.peak>1?` × ${inp.peak} peak = ${fmt(users)}`:""} · ${fmt(inp.isl)} in / ${fmt(inp.osl)} out tokens · ${tpsLine}</p>
        ${brandHtml()}
      </div>
      <div class="noprint btnrow">
        <button class="ghost sm" id="btnPrint">Print / Save PDF</button>
        <button class="ghost sm" id="btnXlsx">Excel (.xlsx)</button>
        <button class="ghost sm" id="btnCsv">CSV</button>
        <button class="ghost sm" id="btnCopy">Copy summary</button>
      </div>
    </div>
    ${recBlock}
    <div class="hero">
      <div class="kpi"><div class="k">Selected GPU</div><div class="v">${sel.infeasible?"n/a":fmt(sel.gpus)}</div><div class="s">${sel.g.name}${sel.spareGpus?` · incl. ${fmt(sel.spareGpus)} spare`:""}</div></div>
      <div class="kpi"><div class="k">Servers</div><div class="v">${fmt(sel.servers)}</div><div class="s">at ${sel.gps} GPUs / server</div></div>
      <div class="kpi"><div class="k">Total GPU memory</div><div class="v">${gbtb(sel.totalGpuMemRaw)}</div><div class="s">${gbtb(sel.memUsedGB)} needed</div></div>
      <div class="kpi"><div class="k">Compute (${precLabel})</div><div class="v">${tf(sel.totalTflops)}</div><div class="s">dense tensor${sel.reqTflops&&isFinite(sel.computeUtil)?` · ~${fmt(sel.computeUtil*100,sel.computeUtil<0.1?1:0)}% used`:""}</div></div>
      <div class="kpi"><div class="k">Host (CPU) RAM</div><div class="v">${ramf(sel.hostRecGB)}</div><div class="s">per server${sel.hostIntegrated?" (Grace/Vera LPDDR5X)":` · min ${ramf(sel.hostMinGB)}`}</div></div>
      <div class="kpi"><div class="k">Speed per user</div><div class="v">${fmt(sel.perUserTps,0)}</div><div class="s">tokens/s (estimated)</div></div>
      <div class="kpi"><div class="k">Time to first token</div><div class="v">${!isFinite(sel.ttft)?"–":sel.ttft<1?fmt(sel.ttft*1000,0)+" ms":fmt(sel.ttft,1)+" s"}</div><div class="s">prefill of ${fmt(inp.isl)} tokens</div></div>
      <div class="kpi"><div class="k">Power</div><div class="v">${fmt(sel.facKW, sel.facKW<10?1:0)} kW</div><div class="s">facility (PUE ${C.pue})</div></div>
      <div class="kpi"><div class="k">Rough hardware cost</div><div class="v" style="font-size:18px">${c?range(c.capLo,c.capHi):"–"}</div><div class="s">indicative only (see caveat)</div></div>
    </div>
    ${W.join("")}
  </div>

  <div class="two">
    <div class="card">
      <h2>Sizing breakdown: ${sel.g.name}</h2>
      <p class="sub">How the GPU count was worked out</p>
      <table>
        <tr><td>Model weights @ ${precLabel}</td><td class="num">${gbtb(sel.weightsGB)}</td></tr>
        <tr><td>KV cache per token${inp.prec==="FP4"?" (FP8 KV)":""}</td><td class="num">${fmt(sel.kvTokB/1024,1)} KiB</td></tr>
        ${sel.stateGB>0?`<tr><td>Fixed recurrent state per user</td><td class="num">${fmt(sel.stateGB*1000,0)} MB</td></tr>`:""}
        <tr><td>KV cache + state per user (${fmt(sel.ctx)} tokens)</td><td class="num">${fmt(sel.kvUserGB*1000,1)} MB</td></tr>
        <tr><td>Users sized for${inp.peak>1?` (${inp.peak}× peak)`:""}</td><td class="num">${fmt(users)}</td></tr>
        <tr><td>KV cache, all users</td><td class="num">${gbtb(sel.kvTotalGB)}</td></tr>
        <tr><td>Usable memory per GPU (${inp.util}% of ${sel.g.mem} GB)</td><td class="num">${fmt(sel.usable,1)} GB</td></tr>
        <tr><td>GPUs per model copy (tensor parallel)</td><td class="num">${fmt(sel.rep)}</td></tr>
        <tr><td>Model copies (replicas)</td><td class="num">${fmt(sel.replicas)}</td></tr>
        <tr><td>GPUs for workload</td><td class="num">${fmt(sel.baseGpus)}</td></tr>
        <tr><td>Spare GPUs (${esc(redundLabel)})</td><td class="num">${fmt(sel.spareGpus)}</td></tr>
        <tr class="total"><td>Total dedicated GPUs</td><td class="num">${sel.infeasible?"n/a":fmt(sel.gpus)}</td></tr>
        <tr><td>Limiting factor</td><td class="num">${sel.limit}</td></tr>
        <tr><td>Aggregate output throughput</td><td class="num">${fmt(sel.aggTps)} tok/s</td></tr>
      </table>
    </div>
    <div class="card">
      <h2>Inputs &amp; model details</h2>
      <p class="sub">What this report is based on</p>
      <table>
        <tr><td>Model</td><td>${m.name} ${m.open?'<span class="tag g">open weights</span>':'<span class="tag b">closed / API only</span>'}${m.est?'<span class="tag w">estimated size</span>':""}${m.edited?'<span class="tag w">edited</span>':""}</td></tr>
        <tr><td>Parameters</td><td>${fmt(m.total,m.total<100?2:0)}B total / ${fmt(m.active,m.active<100?2:0)}B active ${m.total!==m.active?"(MoE)":"(dense)"}</td></tr>
        <tr><td>Notes</td><td class="muted">${m.note}</td></tr>
        <tr><td>Application</td><td>${$("usecase").querySelector(`option[value="${inp.usecase}"]`).textContent}</td></tr>
        <tr><td>Precision</td><td>${precLabel}</td></tr>
        <tr><td>Input / output length</td><td>${fmt(inp.isl)} / ${fmt(inp.osl)} tokens</td></tr>
        <tr><td>Concurrent users / peak factor</td><td>${fmt(inp.users)} × ${inp.peak}</td></tr>
        <tr><td>Performance target</td><td>${tpsLine}</td></tr>
        <tr><td>Max memory utilisation</td><td>${inp.util}%</td></tr>
        <tr><td>GPUs per server</td><td>${inp.gps}</td></tr>
        <tr><td>Redundancy</td><td>${esc(redundLabel)}</td></tr>
      </table>
    </div>
  </div>

  ${sel.infeasible?"":`
  <div class="card">
    <h2>Hardware specification: ${sel.g.name} deployment</h2>
    <p class="sub">Bill-of-materials view for the selected GPU</p>
    <div class="two">
      <table>
        <tr><th colspan="2">GPU</th></tr>
        <tr><td>GPU model / architecture</td><td class="num">${sel.g.name} · ${sel.g.arch} · ${sel.g.form}</td></tr>
        <tr><td>Dedicated GPUs (incl. spares)</td><td class="num"><b>${fmt(sel.gpus)}</b></td></tr>
        <tr><td>Memory per GPU</td><td class="num">${sel.g.memRaw||sel.g.mem} GB${sel.g.memRaw?` (${sel.g.mem} GB usable)`:""}</td></tr>
        <tr><td><b>Total GPU memory capacity</b></td><td class="num"><b>${gbtb(sel.totalGpuMemRaw)}</b></td></tr>
        <tr><td>Usable at ${inp.util}% / required</td><td class="num">${gbtb(sel.totalGpuMemUsable)} / ${gbtb(sel.memUsedGB)}</td></tr>
        <tr><td>Memory bandwidth per GPU / total</td><td class="num">${fmt(sel.g.bw/1000,2)} / ${fmt(sel.totalBW,1)} TB/s</td></tr>
        <tr><th colspan="2">Compute (dense tensor TFLOPS)</th></tr>
        <tr><td>Per GPU: FP16 · FP8/INT8 · FP4 · TF32</td><td class="num">${fmt(sel.g.fp16)} · ${fmt(sel.g.fp8)} · ${sel.g.fp4?fmt(sel.g.fp4):"n/a"} · ${fmt(sel.g.fp32)}</td></tr>
        <tr><td>Total FP16 / BF16</td><td class="num">${tf(sel.totalTflops16)}</td></tr>
        <tr><td>Total FP8 / INT8</td><td class="num">${tf(sel.totalTflops8)}</td></tr>
        <tr><td>Total FP4</td><td class="num">${sel.g.fp4?tf(sel.totalTflops4):"not supported"}</td></tr>
        <tr><td>Total FP32 (TF32)</td><td class="num">${tf(sel.totalTflops32)}</td></tr>
        <tr><td><b>Total at selected precision</b></td><td class="num"><b>${tf(sel.totalTflops)}</b></td></tr>
        <tr><td>Effective compute needed for target</td><td class="num">${sel.reqTflops?tf(sel.reqTflops)+` (${fmt(sel.computeUtil*100,sel.computeUtil<0.1?1:0)}%)`:"n/a"}</td></tr>
      </table>
      <table>
        <tr><th colspan="2">Servers &amp; host</th></tr>
        <tr><td>Servers</td><td class="num"><b>${fmt(sel.servers)}</b> × ${sel.gps} GPUs</td></tr>
        <tr><td>GPU memory per server</td><td class="num">${gbtb(sel.gpuMemPerSrv)}</td></tr>
        <tr><td><b>Host (CPU) RAM per server${sel.hostIntegrated?"":": recommended"}</b></td><td class="num"><b>${ramf(sel.hostRecGB)}</b></td></tr>
        ${sel.hostIntegrated?`<tr><td>Host memory type</td><td class="num">Grace/Vera LPDDR5X (integrated)</td></tr>`:`<tr><td>Host (CPU) RAM per server: minimum</td><td class="num">${ramf(sel.hostMinGB)}</td></tr>`}
        <tr><td>Host RAM, all servers</td><td class="num">${ramf(sel.hostTotalRecGB)}</td></tr>
        <tr><td>CPU cores per server (suggested)</td><td class="num">≥ ${fmt(sel.cpuCores)} physical cores</td></tr>
        <tr><td>Local NVMe per server (suggested)</td><td class="num">≥ ${gbtb(Math.max(1920, sel.weightsGB*2))}</td></tr>
        <tr><th colspan="2">Power</th></tr>
        <tr><td>GPU power (TDP)</td><td class="num">${fmt(sel.gpuKW,1)} kW</td></tr>
        <tr><td>IT load incl. host (+${C.hostOverhead*100}%)</td><td class="num">${fmt(sel.itKW,1)} kW</td></tr>
        <tr><td>Facility power (PUE ${C.pue})</td><td class="num">${fmt(sel.facKW,1)} kW</td></tr>
      </table>
    </div>
    <p class="muted" style="font-size:12.5px;margin-top:12px">Host RAM rule: about 0.85–1× the GPU memory per server at minimum, and 2× recommended (capped at 4 TB per 8 GPUs). NVIDIA's DGX B300 ships with 2 TB of system memory (configurable to 4 TB) for 2.1 TB of GPU memory.${m.id==="deepseek-v41"?" DeepSeek V4.1's 196B-parameter Engram tables can be placed in host RAM; budget about 200 GB (FP8) to 400 GB (BF16) extra.":""}</p>
  </div>`}

  ${costCard}

  <div class="card">
    <h2>All GPU options compared</h2>
    <p class="sub">Ranked by: ${$("rankBy").selectedOptions[0].textContent}. The highlighted row is your selected GPU. Hardware cost is a rough range.</p>
    <div class="scroll"><table>
      <thead><tr><th>GPU</th><th class="num">Memory</th><th class="num">GPUs / copy</th><th class="num">Total GPUs</th><th></th><th class="num">Servers</th><th class="num">Total GPU mem</th><th class="num">Total compute</th><th class="num">Host RAM / server</th><th class="num">Facility power</th><th class="num">Rough hardware cost</th><th>Limited by</th></tr></thead>
      <tbody>${rows}</tbody>
    </table></div>
  </div>

  ${methodHtml()}`;
  const rep=$("report"); rep.innerHTML=html; rep.style.display="block";
  rep.scrollIntoView({behavior:"smooth"});
  $("btnPrint").onclick=printReport;
  $("btnXlsx").onclick=()=>exportXlsx("single");
  $("btnCsv").onclick=exportCsv;
  $("btnCopy").onclick=copySummary;
}
function methodHtml(){
  return `<div class="card">
    <h2>Method &amp; assumptions</h2>
    <ul class="notes">
      <li><b>Weights</b> = total parameters × bytes per parameter (FP32 4 B, FP16/BF16 2 B, FP8/INT8 1 B, FP4 0.5 B). Mixture-of-Experts (MoE) models hold <i>all</i> experts in GPU memory.</li>
      <li><b>KV cache</b> = per-token KV × (input + output tokens) × users, plus any fixed per-user recurrent state (linear-attention models). FP4 runs assume an FP8 KV cache. A ${C.overhead*100}% runtime overhead is added.</li>
      <li><b>Peak load factor</b> multiplies concurrent users before sizing. <b>Redundancy</b> adds spare GPUs (N+1 = one extra server's worth, rounded to whole model copies).</li>
      <li><b>GPUs per copy</b> is the smallest power-of-two tensor-parallel group that holds the weights (widened if needed to reach the speed target). Larger groups are also tried, and the cheapest total is kept.</li>
      <li><b>Token speed</b>: decoding is bandwidth-bound at ${C.bwEff*100}% effective HBM bandwidth. Compute is capped at ${C.mfuDecode*100}% of dense TFLOPS (decode) and ${C.mfuPrefill*100}% (prefill). GPUs without native FP4 compute 4-bit weights at FP8 (or FP16) rate.</li>
      <li><b>Compute</b> figures are dense (non-sparse) tensor TFLOPS. FP32 = TF32 (AMD MI355X: FP32 matrix).</li>
      <li><b>GPUs per server</b> is capped per form factor (8 for PCIe/SXM/OAM servers, 72 for NVL72 racks) so comparisons stay realistic.</li>
      <li><b>Host RAM</b>: minimum ≈0.85×, recommended 2× GPU memory per server (capped at 4 TB per 8 GPUs). NVL72 racks use integrated Grace/Vera LPDDR5X. <b>CPU</b>: ≈${C.coresPerGPU} cores per GPU.</li>
      <li><b>Power</b> = GPU TDP × count, +${C.hostOverhead*100}% host/network/storage, × PUE ${C.pue}.</li>
      <li><b>Cost</b>: rough ranges only (see caveat). GPU prices are editable. Server base cost per GPU slot by form factor, +10–20% network/integration, energy at 60–85% of facility power, plus maintenance.</li>
      <li>Not modelled: prefix caching, speculative decoding, disaggregated prefill/decode, multi-LoRA, request queueing.</li>
      <li>Data last reviewed ${DATA_DATE}. Closed-model sizes and preliminary GPUs (Rubin) are estimates. Edit the <code>MODELS</code> / <code>GPUS</code> tables at the top of the script to update them.</li>
    </ul>
  </div>`;
}

/* ---------------- scenarios ---------------- */
let SCEN = store.get("gpuPlannerScenarios", []);
function saveScenarios(){ store.set("gpuPlannerScenarios", SCEN); renderScenarios(); }
function scLabel(i){ const m=MODELS.find(x=>x.id===i.model), g=GPUS.find(x=>x.id===i.gpu); return {m:m?m.name:i.model, g:g?g.name:i.gpu}; }
function renderScenarios(){
  const t=$("scTable");
  if(!SCEN.length){ t.innerHTML=`<tr><td class="muted">No saved scenarios yet. Set up the inputs above, give it a name, and click <b>Save as scenario</b>. Save one per workload (e.g. "Chatbot", "Code assistant") to build a plan.</td></tr>`; return; }
  t.innerHTML = `<thead><tr><th><input type="checkbox" id="scAll"></th><th>Name</th><th>Model</th><th>Precision</th><th class="num">Users</th><th class="num">In / Out</th><th>GPU</th><th class="num">GPUs/server</th><th>Redundancy</th><th></th></tr></thead><tbody>`+
    SCEN.map((s,i)=>{ const L=scLabel(s.inp); return `<tr><td><input type="checkbox" class="scChk" data-i="${i}"></td><td><b>${esc(s.name)}</b></td><td>${esc(L.m)}</td><td>${PREC_LABEL[s.inp.prec]}</td><td class="num">${fmt(s.inp.users)}${s.inp.peak>1?` ×${s.inp.peak}`:""}</td><td class="num">${fmt(s.inp.isl)} / ${fmt(s.inp.osl)}</td><td>${esc(L.g)}</td><td class="num">${s.inp.gps}</td><td>${({none:"None",n1:"N+1",n2:"N+2",p10:"+10%",p20:"+20%","2n":"2N"})[s.inp.redund||"none"]}</td>
      <td class="btnrow"><button class="ghost sm" data-load="${i}">Load</button><button class="ghost sm" data-up="${i}">Update</button></td></tr>`; }).join("")+"</tbody>";
  $("scAll").onchange=e=>document.querySelectorAll(".scChk").forEach(c=>c.checked=e.target.checked);
  t.querySelectorAll("[data-load]").forEach(b=>b.onclick=()=>{ applyInputs(SCEN[+b.dataset.load].inp); $("scName").value=SCEN[+b.dataset.load].name; toast(`Loaded "${SCEN[+b.dataset.load].name}"`); window.scrollTo({top:0,behavior:"smooth"}); });
  t.querySelectorAll("[data-up]").forEach(b=>b.onclick=()=>{ const s=SCEN[+b.dataset.up]; s.inp=readInputs(); saveScenarios(); toast(`Updated "${s.name}" with current inputs`); });
}
function selectedScenarios(){ return [...document.querySelectorAll(".scChk:checked")].map(c=>SCEN[+c.dataset.i]); }
$("btnSave").onclick=()=>{
  const inp=readInputs(); const L=scLabel(inp);
  const name=$("scName").value.trim() || `${L.m} · ${fmt(inp.users)} users · ${L.g}`;
  const ex=SCEN.find(s=>s.name===name);
  if(ex){ ex.inp=inp; toast(`Updated scenario "${name}"`); } else { SCEN.push({name, inp, saved:new Date().toISOString()}); toast(`Saved scenario "${name}"`); }
  saveScenarios(); $("scName").value="";
};
$("btnDel").onclick=()=>{ const sel=new Set(selectedScenarios()); if(!sel.size){toast("Tick one or more scenarios first");return;} SCEN=SCEN.filter(s=>!sel.has(s)); saveScenarios(); toast(`Deleted ${sel.size} scenario(s)`); };
$("btnExpJson").onclick=()=>{ if(!SCEN.length){toast("No scenarios to export");return;} download("pangolin-gpu-planner-scenarios.json", new Blob([JSON.stringify({version:2, exported:new Date().toISOString(), scenarios:SCEN},null,2)],{type:"application/json"})); };
$("impJson").onchange=e=>{
  const f=e.target.files[0]; if(!f) return;
  const rd=new FileReader(); rd.onload=()=>{ try{ const d=JSON.parse(rd.result); const list=(d.scenarios||d).filter(s=>s&&s.inp&&s.name&&MODELS.some(m=>m.id===s.inp.model)&&GPUS.some(g=>g.id===s.inp.gpu));
      list.forEach(s=>{ const ex=SCEN.find(x=>x.name===s.name); if(ex) ex.inp=s.inp; else SCEN.push(s); }); saveScenarios(); toast(`Imported ${list.length} scenario(s)`);
    }catch(err){ toast("Could not read that file: not a planner scenarios file"); } e.target.value=""; };
  rd.readAsText(f);
};

/* ---------------- compare ---------------- */
let LAST_MULTI = null;
$("btnCompare").onclick=()=>{
  const list=selectedScenarios(); if(list.length<2){toast("Tick 2–6 scenarios to compare");return;}
  const res=list.slice(0,6).map(s=>({s, R:computeAll(s.inp)}));
  const rowsDef = [
    ["Model", x=>x.R.m.name], ["Precision", x=>PREC_LABEL[x.s.inp.prec]], ["Users (× peak)", x=>`${fmt(x.s.inp.users)}${x.s.inp.peak>1?" × "+x.s.inp.peak:""}`],
    ["Input / output tokens", x=>`${fmt(x.s.inp.isl)} / ${fmt(x.s.inp.osl)}`], ["Speed target", x=>x.s.inp.tps?x.s.inp.tps+" tok/s":"none"],
    ["GPU", x=>x.R.sel.g.name], ["GPUs (incl. spares)", x=>x.R.sel.infeasible?"n/a":fmt(x.R.sel.gpus)], ["Servers", x=>fmt(x.R.sel.servers)+" × "+x.R.sel.gps],
    ["Total GPU memory", x=>gbtb(x.R.sel.totalGpuMemRaw)], ["Compute (selected precision)", x=>tf(x.R.sel.totalTflops)],
    ["Host RAM / server", x=>ramf(x.R.sel.hostRecGB)], ["Speed per user", x=>fmt(x.R.sel.perUserTps)+" tok/s"], ["Time to first token", x=>!isFinite(x.R.sel.ttft)?"–":x.R.sel.ttft<1?fmt(x.R.sel.ttft*1000)+" ms":fmt(x.R.sel.ttft,1)+" s"],
    ["Facility power", x=>fmt(x.R.sel.facKW,1)+" kW"], ["Limited by", x=>x.R.sel.limit],
    ["Rough hardware cost", x=>x.R.sel.cost?range(x.R.sel.cost.capLo,x.R.sel.cost.capHi):"–"],
    ["Rough TCO", x=>x.R.sel.cost?range(x.R.sel.cost.tcoLo,x.R.sel.cost.tcoHi)+` (${x.R.sel.cost.years} yr)`:"–"],
    ["Rough cost / 1M tokens", x=>x.R.sel.cost?`${money(x.R.sel.cost.perMLo)} – ${money(x.R.sel.cost.perMHi)}`:"–"],
    ["Recommended GPU for this scenario", x=>x.R.best.infeasible?"none feasible":`${x.R.best.g.name} × ${fmt(x.R.best.gpus)}`]
  ];
  LAST_MULTI = {type:"compare", head:["Metric",...res.map(x=>x.s.name)], rows: rowsDef.map(([k,f])=>[k,...res.map(x=>String(f(x)).replace(/<[^>]+>/g,""))])};
  const html = `<div class="card">
    <div class="rpt-head"><div><h2>Pangolin GPU Planner: scenario comparison</h2>${brandHtml()}</div>
    <div class="noprint btnrow"><button class="ghost sm" id="mPrint">Print / Save PDF</button><button class="ghost sm" id="mXlsx">Excel (.xlsx)</button><button class="ghost sm" id="mCsv">CSV</button><button class="ghost sm" id="mClose">Close</button></div></div>
    <div class="scroll" style="margin-top:12px"><table><thead><tr>${LAST_MULTI.head.map((h,i)=>`<th${i?' class="num"':""}>${esc(h)}</th>`).join("")}</tr></thead>
    <tbody>${LAST_MULTI.rows.map(r=>`<tr>${r.map((c,i)=>`<td${i?' class="num"':""}>${i?esc(c):`<b>${esc(c)}</b>`}</td>`).join("")}</tr>`).join("")}</tbody></table></div>
    <p class="muted" style="font-size:12px;margin-top:10px">Cost rows are rough indicative ranges. See the caveat in any single report.</p></div>`;
  showMulti(html);
};

/* ---------------- multi-workload plan ---------------- */
$("btnPlan").onclick=()=>{
  const list=selectedScenarios(); if(!list.length){toast("Tick one or more scenarios (one per workload) to build a plan");return;}
  const res=list.map(s=>({s, R:computeAll(s.inp)}));
  const ok=res.filter(x=>!x.R.sel.infeasible);
  const sum=(f)=>ok.reduce((a,x)=>a+(f(x)||0),0);
  // consolidate by GPU type
  const byGpu={};
  ok.forEach(x=>{ const g=x.R.sel.g, k=g.id; byGpu[k]=byGpu[k]||{g,gpus:0,servers:0,mem:0,tflops16:0,tflops8:0,host:0,kw:0,capLo:0,capHi:0,tcoLo:0,tcoHi:0,n:0};
    const b=byGpu[k], r=x.R.sel; b.n++; b.gpus+=r.gpus; b.servers+=r.servers; b.mem+=r.totalGpuMemRaw; b.tflops16+=r.totalTflops16; b.tflops8+=r.totalTflops8; b.host+=r.hostTotalRecGB; b.kw+=r.facKW;
    if(r.cost){ b.capLo+=r.cost.capLo; b.capHi+=r.cost.capHi; b.tcoLo+=r.cost.tcoLo; b.tcoHi+=r.cost.tcoHi; } });
  // single-GPU-type alternatives for the whole plan
  const alt = GPUS.map(g=>{ let gpus=0, servers=0, kw=0, lo=0, hi=0, bad=false;
      res.forEach(x=>{ const r=x.R.all.find(a=>a.g.id===g.id); if(r.infeasible) bad=true; else { gpus+=r.gpus; servers+=r.servers; kw+=r.facKW; if(r.cost){lo+=r.cost.capLo; hi+=r.cost.capHi;} } });
      return {g,gpus,servers,kw,lo,hi,bad}; })
    .sort((a,b)=>(a.bad-b.bad)||(a.servers-b.servers)||(a.kw-b.kw));
  const wl = res.map(x=>{ const r=x.R.sel; return [x.s.name, x.R.m.name, PREC_LABEL[x.s.inp.prec], fmt(x.s.inp.users)+(x.s.inp.peak>1?" ×"+x.s.inp.peak:""), r.g.name,
      r.infeasible?"n/a":fmt(r.gpus), fmt(r.servers), gbtb(r.totalGpuMemRaw), tf(r.totalTflops), ramf(r.hostRecGB), r.infeasible?"–":fmt(r.facKW,1)+" kW", r.cost?range(r.cost.capLo,r.cost.capHi):"–", r.infeasible?r.limit:r.limit]; });
  const T = {gpus:sum(x=>x.R.sel.gpus), servers:sum(x=>x.R.sel.servers), mem:sum(x=>x.R.sel.totalGpuMemRaw), host:sum(x=>x.R.sel.hostTotalRecGB), kw:sum(x=>x.R.sel.facKW),
             capLo:sum(x=>x.R.sel.cost&&x.R.sel.cost.capLo), capHi:sum(x=>x.R.sel.cost&&x.R.sel.cost.capHi), tcoLo:sum(x=>x.R.sel.cost&&x.R.sel.cost.tcoLo), tcoHi:sum(x=>x.R.sel.cost&&x.R.sel.cost.tcoHi)};
  const head=["Workload","Model","Precision","Users","GPU","GPUs","Servers","GPU memory","Compute","Host RAM/server","Facility power","Rough hardware cost","Limited by"];
  LAST_MULTI = {type:"plan", head, rows:[...wl, ["TOTAL","","","","",fmt(T.gpus),fmt(T.servers),gbtb(T.mem),"",ramf(T.host)+" total",fmt(T.kw,1)+" kW",range(T.capLo,T.capHi),""]],
     byGpu:Object.values(byGpu).map(b=>[b.g.name,b.n,b.gpus,b.servers,gbtb(b.mem),tf(b.tflops16),tf(b.tflops8),ramf(b.host),fmt(b.kw,1)+" kW",range(b.capLo,b.capHi),range(b.tcoLo,b.tcoHi)]),
     alt:alt.map(a=>[a.g.name, a.bad?"not feasible for all":fmt(a.gpus), a.bad?"–":fmt(a.servers), a.bad?"–":fmt(a.kw,1)+" kW", a.bad?"–":range(a.lo,a.hi)])};
  const bad=res.filter(x=>x.R.sel.infeasible);
  const bestAlt=alt.find(a=>!a.bad);
  const html = `<div class="card">
    <div class="rpt-head"><div><h2>Pangolin GPU Planner deployment plan: ${res.length} workload${res.length>1?"s":""}</h2>${brandHtml()}</div>
    <div class="noprint btnrow"><button class="ghost sm" id="mPrint">Print / Save PDF</button><button class="ghost sm" id="mXlsx">Excel (.xlsx)</button><button class="ghost sm" id="mCsv">CSV</button><button class="ghost sm" id="mClose">Close</button></div></div>
    <div class="hero" style="margin-top:12px">
      <div class="kpi"><div class="k">Total GPUs</div><div class="v">${fmt(T.gpus)}</div><div class="s">${Object.keys(byGpu).length} GPU type(s)</div></div>
      <div class="kpi"><div class="k">Servers</div><div class="v">${fmt(T.servers)}</div><div class="s">dedicated per workload</div></div>
      <div class="kpi"><div class="k">Total GPU memory</div><div class="v">${gbtb(T.mem)}</div></div>
      <div class="kpi"><div class="k">Host RAM (all servers)</div><div class="v">${ramf(T.host)}</div></div>
      <div class="kpi"><div class="k">Facility power</div><div class="v">${fmt(T.kw,T.kw<10?1:0)} kW</div></div>
      <div class="kpi"><div class="k">Rough hardware cost</div><div class="v" style="font-size:18px">${range(T.capLo,T.capHi)}</div><div class="s">indicative only</div></div>
      <div class="kpi"><div class="k">Rough TCO</div><div class="v" style="font-size:18px">${range(T.tcoLo,T.tcoHi)}</div><div class="s">per-scenario cost settings</div></div>
    </div>
    ${bad.length?`<div class="badbox">${bad.map(x=>`<b>${esc(x.s.name)}</b>: ${esc(x.R.sel.limit)} on ${x.R.sel.g.name}`).join("<br>")}. These workloads are left out of the totals.</div>`:""}
    <h3 style="font-size:14px;margin:14px 0 6px">Workloads (each on its own selected GPU)</h3>
    <div class="scroll"><table><thead><tr>${head.map((h,i)=>`<th${i>4&&i<12?' class="num"':""}>${h}</th>`).join("")}</tr></thead>
      <tbody>${LAST_MULTI.rows.map((r,ri)=>`<tr class="${ri===LAST_MULTI.rows.length-1?"total":""}">${r.map((c,i)=>`<td${i>4&&i<12?' class="num"':""}>${esc(c)}</td>`).join("")}</tr>`).join("")}</tbody></table></div>
    <h3 style="font-size:14px;margin:18px 0 6px">Consolidated bill of materials by GPU type</h3>
    <div class="scroll"><table><thead><tr><th>GPU</th><th class="num">Workloads</th><th class="num">GPUs</th><th class="num">Servers</th><th class="num">GPU memory</th><th class="num">FP16 compute</th><th class="num">FP8 compute</th><th class="num">Host RAM</th><th class="num">Power</th><th class="num">Rough hardware</th><th class="num">Rough TCO</th></tr></thead>
      <tbody>${LAST_MULTI.byGpu.map(r=>`<tr>${r.map((c,i)=>`<td${i?' class="num"':""}>${esc(typeof c==="number"?fmt(c):c)}</td>`).join("")}</tr>`).join("")}</tbody></table></div>
    <p class="muted" style="font-size:12px">Workloads on the same GPU type are sized on separate servers (conservative). Sharing servers or using multi-model serving can reduce the server count.</p>
    <h3 style="font-size:14px;margin:18px 0 6px">What if the whole plan used one GPU type?</h3>
    ${bestAlt?`<div class="rec"><h3>Best single GPU type for this plan: ${bestAlt.g.name}</h3><div>${fmt(bestAlt.gpus)} GPUs · ${fmt(bestAlt.servers)} servers · ≈${fmt(bestAlt.kw,1)} kW · rough hardware ${range(bestAlt.lo,bestAlt.hi)}</div></div>`:""}
    <div class="scroll"><table><thead><tr><th>GPU</th><th class="num">GPUs</th><th class="num">Servers</th><th class="num">Facility power</th><th class="num">Rough hardware cost</th></tr></thead>
      <tbody>${LAST_MULTI.alt.map((r,i)=>`<tr class="${i===0&&bestAlt?"top":""}">${r.map((c,j)=>`<td${j?' class="num"':""}>${esc(c)}</td>`).join("")}</tr>`).join("")}</tbody></table></div>
  </div>
  <div class="card">${CAVEAT_HTML}<p class="muted" style="font-size:12.5px;margin:0">Each workload uses the GPU, precision, redundancy and cost settings saved in its scenario.</p></div>`;
  showMulti(html);
};
function showMulti(html){
  const el=$("multi"); el.innerHTML=html; el.style.display="block"; $("report").style.display="none";
  el.scrollIntoView({behavior:"smooth"});
  $("mClose").onclick=()=>{ el.style.display="none"; };
  $("mPrint").onclick=printReport;
  $("mXlsx").onclick=()=>exportXlsx("multi");
  $("mCsv").onclick=()=>{ const M=LAST_MULTI; download(`pangolin-gpu-planner-${M.type}.csv`, new Blob([toCsv([M.head,...M.rows])],{type:"text/csv"})); };
}

/* ---------------- exports ---------------- */
function printReport(){ try{ window.print(); }catch(e){ toast("Printing isn't available here. Open the HTML file directly in your browser."); } }
function toCsv(rows){ return "﻿"+rows.map(r=>r.map(c=>{ const s=String(c??""); return /[",\n]/.test(s)?`"${s.replace(/"/g,'""')}"`:s; }).join(",")).join("\r\n"); }
function compareRows(R){
  return [["Rank","GPU","Vendor","Architecture","Form","Memory GB","GPUs per copy","Total GPUs","Spare GPUs","Servers","Total GPU memory GB","Total TFLOPS (selected precision)","Host RAM per server GB","Facility kW","Rough hardware low $","Rough hardware high $","Rough TCO low $","Rough TCO high $","Limited by"],
    ...R.ranked.map((r,i)=>[i+1,r.g.name,r.g.vendor,r.g.arch,r.g.form,r.g.memRaw||r.g.mem, r.infeasible?"":r.rep, r.infeasible?"n/a":r.gpus, r.spareGpus||0, r.infeasible?"":r.servers,
      r.infeasible?"":r.totalGpuMemRaw, r.infeasible?"":Math.round(r.totalTflops), r.infeasible?"":r.hostRecGB, r.infeasible?"":+r.facKW.toFixed(2),
      r.cost?Math.round(r.cost.capLo):"", r.cost?Math.round(r.cost.capHi):"", r.cost?Math.round(r.cost.tcoLo):"", r.cost?Math.round(r.cost.tcoHi):"", r.limit])];
}
function exportCsv(){ if(!LAST) return; download(`pangolin-gpu-planner-${LAST.m.name.replace(/[^\w.-]+/g,"_")}.csv`, new Blob([toCsv(compareRows(LAST))],{type:"text/csv"})); }
function summaryRows(R){
  const {m,inp,sel,best}=R, c=sel.cost, b=brand();
  return [["Pangolin GPU Planner – AI Inference Capacity & Cost Sizing"],["Company",b.company],["Project",b.project],["Prepared by",b.author],["Generated",R.when.toLocaleString()],["Data last reviewed",DATA_DATE],[],
    ["INPUTS"],["Model",m.name],["Total / active parameters (B)",m.total+" / "+m.active],["Open weights",m.open?"yes":"no (API only)"],["Precision",PREC_LABEL[inp.prec]],
    ["Input / output tokens",inp.isl+" / "+inp.osl],["Concurrent users",inp.users],["Peak factor",inp.peak],["Users sized",sel.users],["Speed target tok/s",inp.tps||"none"],
    ["GPU",sel.g.name],["Memory utilisation %",inp.util],["GPUs per server",inp.gps],["Redundancy",inp.redund],[],
    ["RESULT – SELECTED GPU"],["Total GPUs",sel.infeasible?"not feasible":sel.gpus],["  of which spare",sel.spareGpus],["Servers",sel.servers],["GPUs per model copy",sel.rep],["Model copies",sel.replicas],
    ["Weights GB",+sel.weightsGB.toFixed(1)],["KV cache + state per user MB",+(sel.kvUserGB*1000).toFixed(1)],["Total GPU memory GB",sel.totalGpuMemRaw],["Memory required GB",+(sel.memUsedGB||0).toFixed(0)],
    ["Total TFLOPS FP16",sel.totalTflops16],["Total TFLOPS FP8/INT8",sel.totalTflops8],["Total TFLOPS FP4",sel.totalTflops4||"n/a"],["Total TFLOPS TF32",sel.totalTflops32],
    ["Host RAM per server GB (recommended)",sel.hostRecGB],["Host RAM per server GB (minimum)",sel.hostMinGB],["CPU cores per server",sel.cpuCores],
    ["Speed per user tok/s",+(sel.perUserTps||0).toFixed(1)],["Time to first token s",+(sel.ttft||0).toFixed(3)],["Facility power kW",+(sel.facKW||0).toFixed(1)],["Limited by",sel.limit],[],
    ["ROUGH COST ESTIMATE – INDICATIVE ONLY, NOT A QUOTE (±30–50% or more)"],
    ["Hardware CapEx low $",c?Math.round(c.capLo):""],["Hardware CapEx high $",c?Math.round(c.capHi):""],["TCO low $ ("+(c?c.years:"")+" yr)",c?Math.round(c.tcoLo):""],["TCO high $",c?Math.round(c.tcoHi):""],
    ["Cost per 1M tokens low $",c?+c.perMLo.toFixed(3):""],["Cost per 1M tokens high $",c?+c.perMHi.toFixed(3):""],[],
    ["RECOMMENDATION"],["Best GPU",best.infeasible?"none feasible":best.g.name],["GPUs",best.infeasible?"":best.gpus],["Servers",best.infeasible?"":best.servers]];
}
function exportXlsx(which){
  const sheets=[];
  if(which==="single"){ if(!LAST) return; sheets.push({name:"Summary",rows:summaryRows(LAST),widths:[42,34]}); sheets.push({name:"GPU comparison",rows:compareRows(LAST),widths:[6,26,9,15,8,10,10,10,9,9,14,16,14,10,14,14,14,14,30]}); }
  else { const M=LAST_MULTI; if(!M) return;
    sheets.push({name:M.type==="plan"?"Plan":"Comparison", rows:[M.head,...M.rows], widths:M.head.map((h,i)=>i?18:28)});
    if(M.byGpu) sheets.push({name:"BOM by GPU", rows:[["GPU","Workloads","GPUs","Servers","GPU memory","FP16 compute","FP8 compute","Host RAM","Power","Rough hardware","Rough TCO"],...M.byGpu], widths:[26,10,8,8,12,14,14,12,10,20,20]});
    if(M.alt) sheets.push({name:"Single GPU options", rows:[["GPU","GPUs","Servers","Facility power","Rough hardware"],...M.alt], widths:[26,12,10,14,22]});
  }
  sheets.push({name:"Notes", rows:[["Caveat emptor"],["All cost figures are rough indicative ranges for early budgeting only. They are not quotes and may be wrong by ±30–50% or more."],
    ["Excluded: facility build-out, cooling plant, software licences, staffing, taxes, financing."],["Data last reviewed: "+DATA_DATE],
    ["Sizing method: weights + KV cache + state, tensor-parallel copies, bandwidth/compute speed checks, redundancy. See the HTML report for details."]], widths:[120]});
  const b=brand(); const fname=`pangolin-gpu-planner-${which==="single"?LAST.m.name:LAST_MULTI.type}${b.project?"-"+b.project:""}`.replace(/[^\w.-]+/g,"_")+".xlsx";
  download(fname, makeXlsx(sheets));
}
/* ---- minimal dependency-free .xlsx writer (stored ZIP + SpreadsheetML) ---- */
const CRC_T=(()=>{const t=new Uint32Array(256);for(let n=0;n<256;n++){let c=n;for(let k=0;k<8;k++)c=c&1?0xEDB88320^(c>>>1):c>>>1;t[n]=c>>>0;}return t;})();
function crc32(u){let c=0xFFFFFFFF;for(let i=0;i<u.length;i++)c=CRC_T[(c^u[i])&255]^(c>>>8);return (c^0xFFFFFFFF)>>>0;}
function zipStore(files){
  const enc=new TextEncoder(), parts=[], cen=[]; let off=0;
  const u16=v=>[v&255,(v>>8)&255], u32=v=>[v&255,(v>>8)&255,(v>>16)&255,(v>>>24)&255];
  for(const f of files){
    const name=enc.encode(f.name), data=typeof f.data==="string"?enc.encode(f.data):f.data, crc=crc32(data);
    const loc=new Uint8Array([...u32(0x04034b50),...u16(20),...u16(0x0800),...u16(0),...u16(0),...u16(0x21),...u32(crc),...u32(data.length),...u32(data.length),...u16(name.length),...u16(0)]);
    parts.push(loc,name,data);
    cen.push(new Uint8Array([...u32(0x02014b50),...u16(20),...u16(20),...u16(0x0800),...u16(0),...u16(0),...u16(0x21),...u32(crc),...u32(data.length),...u32(data.length),...u16(name.length),...u16(0),...u16(0),...u16(0),...u16(0),...u32(0),...u32(off)]),name);
    off+=loc.length+name.length+data.length;
  }
  const cenSize=cen.reduce((a,p)=>a+p.length,0);
  const end=new Uint8Array([...u32(0x06054b50),...u16(0),...u16(0),...u16(files.length),...u16(files.length),...u32(cenSize),...u32(off),...u16(0)]);
  return new Blob([...parts,...cen,end],{type:"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"});
}
function makeXlsx(sheets){
  const x=s=>String(s).replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c])).replace(/[\u0000-\u0008\u000B\u000C\u000E-\u001F]/g,"");
  const col=i=>{let s="";i++;while(i>0){const m=(i-1)%26;s=String.fromCharCode(65+m)+s;i=Math.floor((i-1)/26);}return s;};
  const sheetXml=(sh)=>{
    const cols=sh.widths?`<cols>${sh.widths.map((w,i)=>`<col min="${i+1}" max="${i+1}" width="${w}" customWidth="1"/>`).join("")}</cols>`:"";
    const rows=sh.rows.map((r,ri)=>`<row r="${ri+1}">${(r||[]).map((v,ci)=>{
      const ref=col(ci)+(ri+1), bold=(ri===0||(r.length===1&&/^[A-Z][A-Z \-–()0-9,.%±]+$/.test(String(v))))?' s="1"':"";
      if(v===""||v===null||v===undefined) return "";
      if(typeof v==="string" && /^-?\d{1,3}(,\d{3})*(\.\d+)?$|^-?\d+(\.\d+)?$/.test(v.trim())) v=Number(v.replace(/,/g,""));
      if(typeof v==="number"&&isFinite(v)) return `<c r="${ref}"${bold}><v>${v}</v></c>`;
      return `<c r="${ref}" t="inlineStr"${bold}><is><t xml:space="preserve">${x(v)}</t></is></c>`; }).join("")}</row>`).join("");
    return `<?xml version="1.0" encoding="UTF-8" standalone="yes"?><worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/></sheetView></sheetViews>${cols}<sheetData>${rows}</sheetData></worksheet>`;
  };
  const files=[
    {name:"[Content_Types].xml",data:`<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/><Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>${sheets.map((s,i)=>`<Override PartName="/xl/worksheets/sheet${i+1}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>`).join("")}</Types>`},
    {name:"_rels/.rels",data:`<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>`},
    {name:"xl/workbook.xml",data:`<?xml version="1.0" encoding="UTF-8" standalone="yes"?><workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets>${sheets.map((s,i)=>`<sheet name="${x(s.name.slice(0,31))}" sheetId="${i+1}" r:id="rId${i+1}"/>`).join("")}</sheets></workbook>`},
    {name:"xl/_rels/workbook.xml.rels",data:`<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">${sheets.map((s,i)=>`<Relationship Id="rId${i+1}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet${i+1}.xml"/>`).join("")}<Relationship Id="rId${sheets.length+1}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/></Relationships>`},
    {name:"xl/styles.xml",data:`<?xml version="1.0" encoding="UTF-8" standalone="yes"?><styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><fonts count="2"><font><sz val="11"/><name val="Calibri"/></font><font><b/><sz val="11"/><name val="Calibri"/></font></fonts><fills count="2"><fill><patternFill patternType="none"/></fill><fill><patternFill patternType="gray125"/></fill></fills><borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders><cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs><cellXfs count="2"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/><xf numFmtId="0" fontId="1" fillId="0" borderId="0" xfId="0" applyFont="1"/></cellXfs><cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles></styleSheet>`},
    ...sheets.map((s,i)=>({name:`xl/worksheets/sheet${i+1}.xml`,data:sheetXml(s)}))
  ];
  return zipStore(files);
}
async function copySummary(){
  const R=LAST; if(!R) return; const {m,inp,sel,best,ranked}=R; const c=sel.cost;
  const txt=`Pangolin GPU Planner – ${m.name}\n${PREC_LABEL[inp.prec]}, ${inp.users} users${inp.peak>1?" x"+inp.peak+" peak":""}, ${inp.isl} in / ${inp.osl} out tokens, ${inp.tps?inp.tps+" tok/s/user":"no speed target"}, redundancy ${inp.redund}\n`+
    (best.infeasible?"No feasible GPU\n":`Recommended: ${best.g.name} x ${best.gpus} GPUs (${best.servers} servers @ ${inp.gps}/server, ~${best.facKW.toFixed(1)} kW)\n`)+
    (sel.infeasible?`Selected ${sel.g.name}: not feasible (${sel.limit})\n`:`Selected ${sel.g.name}: ${sel.gpus} GPUs, ${sel.servers} servers | GPU memory ${gbtb(sel.totalGpuMemRaw)} | ${tf(sel.totalTflops)} | host RAM ${ramf(sel.hostRecGB)}/server | ${sel.facKW.toFixed(1)} kW\n`)+
    (c?`Rough hardware cost ${range(c.capLo,c.capHi)}, ${c.years}-yr TCO ${range(c.tcoLo,c.tcoHi)} (INDICATIVE ONLY – not a quote)\n`:"")+"\n"+
    ranked.map((r,i)=>r.infeasible?`${i+1}. ${r.g.name}: not feasible`:`${i+1}. ${r.g.name}: ${r.gpus} GPUs, ${r.servers} servers, ${r.facKW.toFixed(1)} kW`).join("\n");
  try{ await navigator.clipboard.writeText(txt); toast("Summary copied"); }catch(e){ toast("Couldn't copy: clipboard not available"); }
}

/* ---------------- wire up ---------------- */
$("btnReset").onclick=setDefaults;
$("btnGen").onclick=generate;
setDefaults();
renderScenarios();

/* ---------------- PWA: service worker + install prompt ---------------- */
if ("serviceWorker" in navigator && /^https?:/.test(location.protocol)) {
  window.addEventListener("load", () => navigator.serviceWorker.register("sw.js").catch(()=>{}));
}
let deferredInstall=null;
window.addEventListener("beforeinstallprompt", e => { e.preventDefault(); deferredInstall=e; const b=$("btnInstall"); if(b) b.style.display="inline-block"; });
$("btnInstall").onclick = async () => { if(!deferredInstall) return; deferredInstall.prompt(); await deferredInstall.userChoice.catch(()=>{}); deferredInstall=null; $("btnInstall").style.display="none"; };
window.addEventListener("appinstalled", () => toast("Pangolin GPU Planner installed"));
