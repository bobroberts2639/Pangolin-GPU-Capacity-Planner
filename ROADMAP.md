# Roadmap – Pangolin GPU Planner

Current release: **v2.0.0** (2026-11-22), live at https://bobroberts2639.github.io/Pangolin-GPU-Capacity-Planner/

## v3 – planned (target: early December 2026)

### 1. Benchmark calibration
- Calibrate the engine's efficiency factors (now fixed: 70% HBM bandwidth, 40% decode compute, 50% prefill) against published results (MLPerf Inference, vLLM / SGLang / TensorRT-LLM benchmarks), per GPU and, where possible, per model.
- Show "modelled vs. measured" with the source and date for each benchmark.
- Keep the Python engine (`python/pangolin_planner.py`) and the web app in step; update `tests/test_engine.py`.

### 2. Facility planning
- Racks needed, kW per rack, rack power density checks (air vs. liquid cooling thresholds).
- Cooling type per GPU (air / direct liquid cooling); NVL72 rack-scale handling (~120–140 kW per rack).
- Network fabric sizing: scale-up (NVLink / Infinity Fabric) and scale-out (InfiniBand / Ethernet, NICs, switch counts).
- Floor space and power/cooling summary for the facility team.

### 3. Latency targets
- Inputs for maximum time to first token (TTFT) and per-token latency, alongside throughput.
- Size for whichever is stricter. Flag when a target cannot be met on a GPU.

### 4. Data management
- Keep model/GPU data in a maintainable source (the user's spreadsheet or CSV/JSON with a `source` and `last_checked` column for every figure).
- Validation script plus CI check. Show a "data last reviewed" date in reports.

### 5. Custom models and GPUs in the app
- Add/edit models and GPUs from the UI (stored per browser, export/import as JSON), clearly marked as user-supplied in reports.

### 6. Monthly data update
- Scheduled task (monthly) that checks for new GPUs, model releases and price changes, updates `js/data.js`, and opens a **pull request for review**. Nothing merges automatically.

## Later (tabled)
- **Rack elevation diagrams**: per-rack front/rear elevations (U positions, servers, switches, PDUs) generated from a plan.
- **Data-centre-friendly schematics**: power and network one-line diagrams and row/pod layouts, exportable (SVG/PDF) for facility and engineering teams.

## Ideas parking lot
- Training / fine-tuning sizing (currently inference only).
- Prefix caching, speculative decoding, disaggregated prefill/decode in the model.
- Cloud rental vs. own-hardware comparison (with the same caveat emptor rules as costs).
