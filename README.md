# Business Entity Resolution — Amazon ML Challenge 2026

Record-linkage between a deduplicated reference source (S1) and two messy
sources (S2, S3). For every S1 entity we must decide the set of matching records
(zero, one, or many). The official metric is **macro-averaged F0.5 measured per
S1 entity**, which makes precision-heavy false merges expensive and makes
correct *singleton* behaviour (predicting nothing when nothing matches) a
first-class part of the score.

The solution is a feature-based LightGBM matcher (MIT-licensed, far below the 8B
parameter cap). No external business data, geocoding, or registries are used.

## Quick start

```bash
pip install -r requirements.txt

# data-root is optional; omit it to auto-detect the dataset
python run_pipeline.py profile             --data-root student_resource
python run_pipeline.py train               --data-root student_resource
python run_pipeline.py validate            --data-root student_resource
python run_pipeline.py predict             --data-root student_resource
python run_pipeline.py validate-submission --data-root student_resource

# everything, plus the submission zip
python run_pipeline.py all --data-root student_resource --zip --team-name <team>
```

`--data-root` should point at the directory containing
`dataset/train/…`, `dataset/test/…`, `utils/validate_submission.py`.
macOS metadata (`__MACOSX`, `._*`, `.DS_Store`) is ignored automatically.

## Dataset discovery

1. `--data-root <path>` (highest priority)
2. `$AMER_DATA_ROOT`
3. auto-detection: cwd, `./student_resource`, `~/Downloads`, `~/Downloads/student_resource`, `/data`, `/workspace`, `/mnt/data`

## Pipeline stages

| Stage | What it does | Key outputs |
|---|---|---|
| `profile` | measures every real statistic of the supplied data | `reports/data_profile.{md,json}` |
| `train` | labels → candidates → features → hard negatives → LightGBM → calibration → threshold/ambiguity search | `models/*`, `experiments/experiment_log.csv`, `reports/{validation_results,threshold_search,calibration_search,feature_importance,candidate_recall,error_analysis}` |
| `validate` | re-scores the saved model on the held-out S1 fold | `reports/validation_results.*` |
| `predict` | test inference, writes the two submission TSVs | `output/matching_results.tsv`, `output/candidate_pairs.tsv` |
| `validate-submission` | runs the official validator + independent structural checks | `reports/submission_validation.json` |
| `all` | profile → train → validate → predict → validate-submission (+ `--zip`) | `reports/run_summary.json`, `<team>_submission.zip` |

## Experiments

```bash
python run_experiments.py ablation --data-root student_resource   # 13-step ladder
python run_experiments.py variants --data-root student_resource   # submission A..E
python run_experiments.py tune     --data-root student_resource --trials 25 --budget 1800
```

## Memory-efficient by construction

* No Cartesian joins, no dense pairwise matrices.
* Ten union blocks produce one merged candidate set per S1; every pair keeps a
  provenance bitmask plus retrieval ranks.
* Inverted indexes are `int32` numpy posting lists; tokens are tuples.
* Rare-token retrieval uses training-only frequency statistics (no test leakage).
* Candidate feature extraction memoises fuzzy bundles on the string pair.
* Entity-level 80/20 split; the split is by S1 entity, never by pair.

## Output contract

`output/matching_results.tsv`

```
source1_entity_id<TAB>matched_entity_ids
S1-00001<TAB>S2-00047,S2-00193,S3-00812
S1-00002<TAB>S3-00004
S1-00003<TAB>
```

`output/candidate_pairs.tsv` holds the exact final candidate set fed to the
matcher for each S1 entity. Every accepted match appears in it.

## Layout

```
├── run_pipeline.py          CLI (profile/train/validate/predict/validate-submission/all)
├── run_experiments.py       ablation / variants / tune
├── build_zip.py             packages <team>_submission.zip
├── config.yaml              all hyper-parameters
├── src/                     pipeline modules (see src/README-comments)
├── models/                  trained LightGBM + calibration bundle
├── experiments/             experiment_log.csv, configs/
├── reports/                 generated reports
└── output/                  matching_results.tsv, candidate_pairs.tsv
```

## Reproducing on the real (large) dataset

The challenge's Source 2 and Source 3 files can be very large. The code is written
for constrained RAM, but the single biggest lever is the blocking frequency caps in
`config.yaml`. Recommended workflow:

1. **Profile first** (streams row counts, never loads everything):
   `python run_pipeline.py profile --data-root student_resource`
   Inspect `reports/data_profile.md` for real row counts, then set the caps.
2. **Tune blocking for your RAM.** Start from the defaults; if candidate volume is
   too high, lower `max_candidates_per_block` (e.g. 100) and the `*_max_postings`
   caps. If candidate recall is below ~0.98, raise `rare_*_token_df` and
   `max_candidates_per_block`.
3. **Train**, checking `reports/candidate_recall.json` for blocking recall and
   `reports/validation_results.json` for macro F0.5.
4. **Predict** and **validate-submission**.

If the box is very small, you can run stages separately and delete intermediate
artefacts between them; the only artefacts needed to resume are in `models/`.

## Reproduction note

The real challenge dataset is distributed separately (it is not part of this
repository). Run the commands above against your extracted `student_resource`
directory; every number in `reports/` will then be a real measurement of the
real data. The `dev/` folder contains a synthetic generator used only to verify
that the pipeline runs end to end in CI-like environments.

## Licenses

LightGBM (MIT) is the primary model; scikit-learn (BSD-3), RapidFuzz (MIT),
Polars (MIT), PyArrow (Apache-2.0), DuckDB (MIT), NumPy (BSD-3), SciPy (BSD-3),
PyYAML (MIT) are the supporting stack. See `requirements.txt`.
