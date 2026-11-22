# Changelog

All notable changes to this project are documented here. Format: [Keep a Changelog](https://keepachangelog.com/), versioning: [SemVer](https://semver.org/).

## [2.0.0] – 2026-11-22
### Added
- **Name:** Pangolin GPU Planner – AI Inference Capacity & Cost Sizing.
- **PWA:** web app manifest, offline service worker, app icons and an Install button.
- **Shared data:** `js/data.js` holds all models, GPUs and assumptions for both the web app and Python.
- **Python and notebook:** Python engine (`python/pangolin_planner.py`), verified identical to the web app (60 random scenarios × 22 GPUs), plus a Jupyter notebook with sweeps, plans, cost sensitivity and Excel export.
- **Precision:** FP4 (native on Blackwell, GB200/GB300, Rubin, MI355X; weight-only 4-bit elsewhere).
- **9 GPUs:** H100 PCIe, H100 NVL, H200 NVL, GB200 NVL72, GB300 NVL72, Rubin (preliminary), MI300X, MI325X, MI355X.
- **Redundancy:** None, N+1, N+2, +10%, +20%, 2N.
- **Load and ranking:** peak load factor (1–3×), and ranking by servers, power, rough cost or GPU count.
- **Rough cost estimate:** CapEx, energy, maintenance, TCO and cost per 1M tokens, with a prominent caveat emptor notice.
- **Scenarios:** save, load, update, delete, import/export (.json), side-by-side comparison, and multi-workload plans with a parts list by GPU type and the best single GPU type for the plan.
- **Exports:** Excel (.xlsx, dependency-free), CSV and print/PDF with branding.
- **Hardware spec:** total GPU memory, TFLOPS by precision, host RAM (minimum/recommended), CPU cores, NVMe, power.
- **Project files:** regression tests, GitHub Actions (Pages deploy and tests), QR codes, README.
### Changed
- GPUs per server capped by form factor (8 PCIe/SXM/OAM, 72 NVL72).
- Copy-size search tries larger tensor-parallel groups and keeps the cheapest total.
### Fixed
- Qwen 3.8-27B KV cache (64 KiB/token, plus DeltaNet state); Kimi K3.1 recurrent state; B300 usable memory (268 GB).
- Infeasible speed targets are now reported instead of running away to millions of GPUs.

## [1.0.0] – 2026-11-21
### Added
- First browser-based planner built from the planner spreadsheet: 7 dropdown inputs, Generate report, comparison of 13 GPUs.
