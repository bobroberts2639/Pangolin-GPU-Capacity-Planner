# Contributing

Thanks for helping improve the Pangolin GPU Planner.

## Updating data (most common)
1. Edit `js/data.js`. Keep everything after `window.PLANNER_DATA =` strict JSON (double quotes, no comments or trailing commas).
2. Cite a source for new or changed figures in the pull request (vendor datasheet or model card preferred). Mark unpublished figures with `"est": true` (models) or `"prelim": true` (GPUs) and explain in `note`.
3. Bump `dataDate`, and bump `CACHE` in `sw.js` so installed apps refresh.
4. Run `pytest -q tests`. If reference values change on purpose, update `tests/test_engine.py` and say why in `CHANGELOG.md`.

## Changing the sizing engine
The engine exists twice: `js/app.js` (`size`, `costEst`, `computeAll`) and `python/pangolin_planner.py`. Change both in the same pull request so they stay identical, and update the reference tests.

## Prices
Prices are deliberately coarse **[low, high] ranges** and must keep the caveat emptor wording wherever costs are shown. Please don't add single-point prices or anything that reads as a quote.

## Style
- The web app stays dependency-free (no build step, works from `file://`).
- Python: standard library for the engine; pandas/matplotlib only in helpers and the notebook.
- Clear notes over clever code: planners are read by people deciding where money goes.

## Reporting issues
Please include the inputs (or an exported scenarios `.json`), what you expected and what you saw.
