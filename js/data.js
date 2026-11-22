/* Pangolin GPU Planner – shared data & assumptions (single source of truth).
 * Used by the web app (index.html) AND the Python engine / notebook (python/pangolin_planner.py).
 * Everything after 'window.PLANNER_DATA =' up to the final ';' must stay STRICT JSON so Python can parse it.
 * Fields: see README.md#data-model. Prices are rough indicative USD ranges – see the cost caveat.
 */
window.PLANNER_DATA = {
  "version": "2.0.0",
  "dataDate": "November 2026",
  "constants": {
    "overhead": 0.1,
    "bwEff": 0.7,
    "mfuDecode": 0.4,
    "mfuPrefill": 0.5,
    "hostOverhead": 0.3,
    "pue": 1.3,
    "hostRamMin": 0.85,
    "hostRamRec": 2,
    "hostRamCapPer8": 4096,
    "coresPerGPU": 16,
    "netLo": 0.1,
    "netHi": 0.2,
    "drawLo": 0.6,
    "drawHi": 0.85
  },
  "bytesPerParam": {
    "FP32": 4,
    "FP16": 2,
    "FP8": 1,
    "FP4": 0.5
  },
  "kvBytes": {
    "FP32": 4,
    "FP16": 2,
    "FP8": 1,
    "FP4": 1
  },
  "precisionLabels": {
    "FP16": "FP16 / BF16",
    "FP8": "FP8 / INT8",
    "FP4": "FP4",
    "FP32": "FP32"
  },
  "ramSteps": [
    64,
    128,
    256,
    384,
    512,
    768,
    1024,
    1536,
    2048,
    3072,
    4096,
    6144,
    8192,
    12288,
    16384
  ],
  "maxGpusPerServer": {
    "PCIe": 8,
    "SXM": 8,
    "OAM": 8,
    "NVL72": 72
  },
  "serverBaseCostPer8": {
    "PCIe": [
      25000,
      60000
    ],
    "SXM": [
      80000,
      150000
    ],
    "OAM": [
      80000,
      150000
    ],
    "NVL72": [
      0,
      0
    ]
  },
  "models": [
    { "id": "mistral7b", "name": "Mistral-7B-v0.3", "total": 7.25, "active": 7.25, "kv": 128, "open": true, "note": "Dense, 32 layers, GQA 8 KV heads." },
    { "id": "llama2-7b", "name": "Llama-2-7B", "total": 6.74, "active": 6.74, "kv": 512, "open": true, "note": "Dense, 32 layers, MHA (no GQA), so it has a large KV cache." },
    { "id": "llama2-13b", "name": "Llama-2-13B", "total": 13, "active": 13, "kv": 800, "open": true, "note": "Dense, 40 layers, MHA." },
    { "id": "llama3-8b", "name": "Llama-3-8B", "total": 8.03, "active": 8.03, "kv": 128, "open": true, "note": "Dense, 32 layers, GQA 8 KV heads." },
    { "id": "llama31-8b", "name": "Llama-3.1-8B", "total": 8.03, "active": 8.03, "kv": 128, "open": true, "note": "Dense, 32 layers, GQA 8 KV heads, 128K context." },
    { "id": "gpt61-terra", "name": "GPT-6.1-Terra", "total": 1200, "active": 60, "kv": 100, "open": false, "est": true, "note": "OpenAI proprietary model, API only." },
    { "id": "gpt62-sol", "name": "GPT-6.2-Sol", "total": 800, "active": 40, "kv": 100, "open": false, "est": true, "note": "OpenAI proprietary model, API only." },
    { "id": "gpt62-astra", "name": "GPT-6.2-Astra", "total": 2000, "active": 100, "kv": 120, "open": false, "est": true, "note": "OpenAI proprietary flagship, API only." },
    { "id": "gpt6-luna", "name": "GPT-6-Luna", "total": 300, "active": 15, "kv": 60, "open": false, "est": true, "note": "OpenAI proprietary model, API only." },
    { "id": "kimi-k31", "name": "Kimi K3.1-Pro", "total": 2780, "active": 104, "kv": 27, "state": 434, "stateEst": true, "open": true, "note": "MoE, 2.78T total / 104B active. 93 layers: 69 KDA linear-attention + 24 gated MLA. The KV cache is small (MLA layers only), but each user also holds a fixed recurrent state in the KDA layers. That state is estimated at ~434 MB (FP32, assumes 96 heads × 128²). Ships natively as MXFP4 weights (~1.4 TB)." },
    { "id": "qwen38-27b", "name": "Qwen 3.8-27B", "total": 27, "active": 27, "kv": 64, "state": 75, "open": true, "note": "Dense 27B, 64 layers: 48 Gated DeltaNet (fixed state) + 16 full-attention layers (4 KV heads × 256 dim), so KV = 64 KiB/token. The DeltaNet state is ≈75 MB per user (BF16)." },
    { "id": "qwen4-max", "name": "Qwen 4-Max", "total": 2728, "active": 71, "kv": 60, "open": false, "kvEst": true, "note": "MoE, 2.4T backbone + 328B N-gram embeddings, 71B active (published by Qwen). API only, no open weights. KV size estimated." },
    { "id": "glm53", "name": "GLM-5.3", "total": 754, "active": 40, "kv": 88, "open": true, "note": "Sized on the published GLM-5/5.1 architecture (754B total / 40B active, MLA-style compressed KV). GLM-5.3 is assumed to be the same." },
    { "id": "deepseek-v41", "name": "DeepSeek V4.1 (Flash)", "total": 748, "active": 16, "kv": 3.125, "open": true, "note": "MoE, 552B backbone + 196B Engram memory, 16B active at decode. Very small KV cache (≈3 KiB/token). The Engram tables can be offloaded to host RAM." },
    { "id": "grok47", "name": "Grok 4.7", "total": 2000, "active": 100, "kv": 120, "open": false, "est": true, "note": "xAI proprietary model, API only." },
    { "id": "muse-ember", "name": "Muse – Ember", "total": 200, "active": 15, "kv": 60, "open": false, "est": true, "note": "Meta proprietary model (Meta Model API)." },
    { "id": "muse-wonder", "name": "Muse – Wonder", "total": 1500, "active": 80, "kv": 100, "open": false, "est": true, "note": "Meta proprietary model. Meta has not disclosed its size." },
    { "id": "muse-wonder-full", "name": "Muse – Wonder-Full", "total": 3000, "active": 150, "kv": 120, "open": false, "est": true, "note": "Meta proprietary model. Meta has not disclosed its size." },
    { "id": "gem4-h", "name": "Gemini 4 – Hydrogen", "total": 150, "active": 10, "kv": 60, "open": false, "est": true, "note": "Google proprietary model, API only." },
    { "id": "gem4-n", "name": "Gemini 4 – Neon", "total": 600, "active": 35, "kv": 80, "open": false, "est": true, "note": "Google proprietary model, API only." },
    { "id": "gem4-c", "name": "Gemini 4 – Carbon", "total": 2500, "active": 120, "kv": 120, "open": false, "est": true, "note": "Google proprietary model, API only." },
    { "id": "claude-haiku", "name": "Claude – Haiku", "total": 200, "active": 20, "kv": 60, "open": false, "est": true, "note": "Anthropic proprietary model, API only." },
    { "id": "claude-sonnet", "name": "Claude – Sonnet", "total": 800, "active": 50, "kv": 100, "open": false, "est": true, "note": "Anthropic proprietary model, API only." },
    { "id": "claude-opus", "name": "Claude – Opus", "total": 2000, "active": 100, "kv": 120, "open": false, "est": true, "note": "Anthropic proprietary model, API only." },
    { "id": "claude-fable", "name": "Claude – Fable", "total": 4000, "active": 200, "kv": 150, "open": false, "est": true, "note": "Anthropic proprietary model, API only." }
  ],
  "gpus": [
    { "id": "a10", "name": "NVIDIA A10", "vendor": "NVIDIA", "arch": "Ampere", "form": "PCIe", "mem": 24, "bw": 600, "fp16": 125, "fp8": 250, "fp4": 0, "fp32": 62, "tdp": 150, "fp8native": false, "price": [2000, 4000] },
    { "id": "a30", "name": "NVIDIA A30", "vendor": "NVIDIA", "arch": "Ampere", "form": "PCIe", "mem": 24, "bw": 933, "fp16": 165, "fp8": 330, "fp4": 0, "fp32": 82, "tdp": 165, "fp8native": false, "price": [4000, 7000] },
    { "id": "a40", "name": "NVIDIA A40", "vendor": "NVIDIA", "arch": "Ampere", "form": "PCIe", "mem": 48, "bw": 696, "fp16": 150, "fp8": 300, "fp4": 0, "fp32": 75, "tdp": 300, "fp8native": false, "price": [4000, 7000] },
    { "id": "a100", "name": "NVIDIA A100 80GB", "vendor": "NVIDIA", "arch": "Ampere", "form": "SXM", "mem": 80, "bw": 2039, "fp16": 312, "fp8": 624, "fp4": 0, "fp32": 156, "tdp": 400, "fp8native": false, "price": [8000, 17000] },
    { "id": "l4", "name": "NVIDIA L4", "vendor": "NVIDIA", "arch": "Ada", "form": "PCIe", "mem": 24, "bw": 300, "fp16": 121, "fp8": 242, "fp4": 0, "fp32": 60, "tdp": 72, "fp8native": true, "price": [2000, 3500] },
    { "id": "l40", "name": "NVIDIA L40", "vendor": "NVIDIA", "arch": "Ada", "form": "PCIe", "mem": 48, "bw": 864, "fp16": 181, "fp8": 362, "fp4": 0, "fp32": 90, "tdp": 300, "fp8native": true, "price": [6000, 9000] },
    { "id": "l40s", "name": "NVIDIA L40S", "vendor": "NVIDIA", "arch": "Ada", "form": "PCIe", "mem": 48, "bw": 864, "fp16": 362, "fp8": 733, "fp4": 0, "fp32": 183, "tdp": 350, "fp8native": true, "price": [7000, 11000] },
    { "id": "h100", "name": "NVIDIA H100 SXM", "vendor": "NVIDIA", "arch": "Hopper", "form": "SXM", "mem": 80, "bw": 3350, "fp16": 989, "fp8": 1979, "fp4": 0, "fp32": 495, "tdp": 700, "fp8native": true, "price": [25000, 40000] },
    { "id": "h100pcie", "name": "NVIDIA H100 PCIe", "vendor": "NVIDIA", "arch": "Hopper", "form": "PCIe", "mem": 80, "bw": 2000, "fp16": 756, "fp8": 1513, "fp4": 0, "fp32": 378, "tdp": 350, "fp8native": true, "price": [22000, 32000] },
    { "id": "h100nvl", "name": "NVIDIA H100 NVL", "vendor": "NVIDIA", "arch": "Hopper", "form": "PCIe", "mem": 94, "bw": 3900, "fp16": 835, "fp8": 1671, "fp4": 0, "fp32": 418, "tdp": 400, "fp8native": true, "price": [25000, 35000], "note": "PCIe card with NVLink bridge (pairs)." },
    { "id": "h200", "name": "NVIDIA H200 SXM", "vendor": "NVIDIA", "arch": "Hopper", "form": "SXM", "mem": 141, "bw": 4800, "fp16": 989, "fp8": 1979, "fp4": 0, "fp32": 495, "tdp": 700, "fp8native": true, "price": [30000, 55000] },
    { "id": "h200nvl", "name": "NVIDIA H200 NVL", "vendor": "NVIDIA", "arch": "Hopper", "form": "PCIe", "mem": 141, "bw": 4800, "fp16": 835, "fp8": 1671, "fp4": 0, "fp32": 418, "tdp": 600, "fp8native": true, "price": [28000, 40000], "note": "PCIe card, 600 W, 2- or 4-way NVLink bridge." },
    { "id": "h800", "name": "NVIDIA H800", "vendor": "NVIDIA", "arch": "Hopper", "form": "SXM", "mem": 80, "bw": 3350, "fp16": 989, "fp8": 1979, "fp4": 0, "fp32": 495, "tdp": 700, "fp8native": true, "price": [25000, 40000], "note": "China-market Hopper with reduced NVLink bandwidth (400 GB/s), so multi-GPU tensor parallelism scales less well." },
    { "id": "b100", "name": "NVIDIA B100", "vendor": "NVIDIA", "arch": "Blackwell", "form": "SXM", "mem": 192, "bw": 8000, "fp16": 1750, "fp8": 3500, "fp4": 7000, "fp32": 875, "tdp": 700, "fp8native": true, "price": [30000, 45000], "note": "Limited availability; most Blackwell volume shipped as B200/GB200." },
    { "id": "b200", "name": "NVIDIA B200", "vendor": "NVIDIA", "arch": "Blackwell", "form": "SXM", "mem": 180, "memRaw": 192, "bw": 8000, "fp16": 2250, "fp8": 4500, "fp4": 9000, "fp32": 1100, "tdp": 1000, "fp8native": true, "price": [30000, 50000], "note": "Planned at 180 GB usable HBM3e (192 GB raw)." },
    { "id": "b300", "name": "NVIDIA B300", "vendor": "NVIDIA", "arch": "Blackwell Ultra", "form": "SXM", "mem": 268, "memRaw": 288, "bw": 8000, "fp16": 2250, "fp8": 4500, "fp4": 15000, "fp32": 1100, "tdp": 1400, "fp8native": true, "price": [40000, 80000], "note": "Planned at 268 GB usable HBM3e (288 GB raw). 1,400 W TDP. Liquid cooling is effectively required. FP4 dense is 15 PFLOPS per NVIDIA (≈13.5 PFLOPS in some independent tables)." },
    { "id": "gb200", "name": "NVIDIA GB200 (NVL72)", "vendor": "NVIDIA", "arch": "Blackwell", "form": "NVL72", "mem": 186, "memRaw": 192, "bw": 8000, "fp16": 2500, "fp8": 5000, "fp4": 10000, "fp32": 1250, "tdp": 1200, "fp8native": true, "price": [45000, 70000], "note": "Rack-scale: 72 GPUs + 36 Grace CPUs in one NVLink domain (~120 kW per rack). The price range is per GPU and includes a share of the rack. Set GPUs per server to 72 to model whole racks." },
    { "id": "gb300", "name": "NVIDIA GB300 (NVL72)", "vendor": "NVIDIA", "arch": "Blackwell Ultra", "form": "NVL72", "mem": 278, "memRaw": 288, "bw": 8000, "fp16": 2500, "fp8": 5000, "fp4": 15000, "fp32": 1250, "tdp": 1400, "fp8native": true, "price": [42000, 90000], "note": "Rack-scale NVL72 (≈20 TB HBM per rack). The per-GPU price includes a rack share (whole racks reported at roughly $3M–$6.5M). Set GPUs per server to 72 to model whole racks." },
    { "id": "rubin", "name": "NVIDIA Rubin (VR NVL72)", "vendor": "NVIDIA", "arch": "Rubin", "form": "NVL72", "mem": 288, "bw": 19200, "fp16": 8750, "fp8": 17500, "fp4": 35000, "fp32": 2000, "tdp": 1800, "fp8native": true, "prelim": true, "price": [60000, 100000], "note": "PRELIMINARY. 288 GB HBM4. Bandwidth revised by NVIDIA to 19.2 TB/s (from 22). FP4 dense = 35 PFLOPS (training figure; 50 PFLOPS inference uses sparsity). The FP8/FP16 figures, the ~1.8 kW power and the price are estimates." },
    { "id": "mi300x", "name": "AMD Instinct MI300X", "vendor": "AMD", "arch": "CDNA 3", "form": "OAM", "mem": 192, "bw": 5300, "fp16": 1307, "fp8": 2615, "fp4": 0, "fp32": 654, "tdp": 750, "fp8native": true, "price": [10000, 20000], "note": "ROCm software stack: check vLLM/SGLang support for your model." },
    { "id": "mi325x", "name": "AMD Instinct MI325X", "vendor": "AMD", "arch": "CDNA 3", "form": "OAM", "mem": 256, "bw": 6000, "fp16": 1307, "fp8": 2615, "fp4": 0, "fp32": 654, "tdp": 1000, "fp8native": true, "price": [15000, 25000], "note": "ROCm software stack: check vLLM/SGLang support for your model." },
    { "id": "mi355x", "name": "AMD Instinct MI355X", "vendor": "AMD", "arch": "CDNA 4", "form": "OAM", "mem": 288, "bw": 8000, "fp16": 2516, "fp8": 5033, "fp4": 10066, "fp32": 157, "tdp": 1400, "fp8native": true, "price": [25000, 40000], "note": "1,400 W, liquid cooled (MI350X is the 1,000 W air-cooled variant). Native MXFP4/FP6. No TF32: FP32 matrix is 157 TFLOPS." }
  ]
};
