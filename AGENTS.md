# AGENTS.md — repository memory

## What this repo is
This repository (previously `amazon_ml_entity_resolution/`) is a competition-ready solution for the Amazon ML
Challenge 2026 Business Entity Resolution task (link S1 reference entities to
S2/S3 records, zero/one/many matches each, scored by macro-averaged per-S1 F0.5).

## Critical facts for future sessions
- **The real dataset is NOT in this sandbox.** It lives on the user's Windows
  machine under `.../6ab10eb3b23ba_student_resource/student_resource/`. Here we
  self-test with a synthetic generator (`dev/make_synthetic.py`) that mirrors the
  schema. Metrics from synthetic data are plumbing checks only — never report them
  as real leaderboard numbers.
- The pipeline is **Python + LightGBM**, feature-based (71 frozen features).
- Entry points: `run_pipeline.py` (profile/train/validate/predict/validate-submission/report/all),
  `run_experiments.py` (ablation/variants/tune), `build_zip.py`.
- `--data-root` is optional; auto-detection looks at `$AMER_DATA_ROOT`, cwd,
  `~/Downloads`, `/workspace`, `/data`, etc.

## How to run in this environment
```bash
cd /workspace/project/ML-Challenge
python dev/make_synthetic.py --out dev/synthetic/student_resource --n-train 1200 --n-test 400
python run_pipeline.py all --data-root dev/synthetic/student_resource --zip --team-name synctest
```
Expect `Submission Validation: PASS` (official validator + internal checks).

## Conventions / gotchas learned
- Feature column order in `src/features.py::FEATURE_NAMES` is **frozen**; models and
  the ablation masks depend on it. When adding features, append and update the
  ablation group lists in `src/ablation.py`.
- Hard-negative mining indexes into the **full** feature matrix (`X[tr_rows]`), so
  ablation runs that pass a column-masked `Xsub` must mine from `X`, then apply the
  same row selection. Getting this wrong raises a LightGBM feature-count mismatch.
- Threshold/source/ambiguity searches must be run on **validation pairs only**
  (`val_pairs`, `probs[val_rows]`). Passing all pairs dilutes the macro score and
  produces absurd drops (e.g. 0.99 → 0.53).
- Country handling is open-set: never hard-code US/India; France must survive.
- Never build a Cartesian product or dense similarity matrix.
- Long runs: use the background pattern
  `(python ... > dev/x.log 2>&1; echo "EXIT=$?" >> dev/x.log) &` then poll the log;
  foreground commands are capped (~1080s).

## Testing
`python -m py_compile src/*.py run_pipeline.py run_experiments.py build_zip.py dev/make_synthetic.py`
then the synthetic `all` run. A healthy run prints `PASS` for both the official and
internal validators.
