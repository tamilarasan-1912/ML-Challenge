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
| `train` | labels → candidates → features → hard negatives → LightGBM → calibration → threshold/ambiguity search | `models/*`, `experiments/experiment_log.csv`, `reports/{validation_results,threshold_search,calibration_search,feature_importance,candidate_recall,error_analysis,block_recall_diagnostic}` |
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

## Architecture Overview (V2 — Memory-Safe)

### Normalization (multi-representation)
Instead of a single normalized string, we generate multiple representations:
- **Name**: light, heavy, core, tokens, core tokens, character n-grams
- **Address**: light, heavy, tokens, numbers, alnum tokens, postal candidates, house number, character n-grams
- **Country**: open-set canonicalization (no hard-coded countries)

### Blocking / Candidate Generation (12 independent blocks)
1. `exact_name` — country + normalized name
2. `core_name` — country + core name (legal suffixes removed)
3. `exact_name_heavy` — country + heavy normalized name
4. `exact_address` — exact normalized address
5. `name_housenumber` — name + house number
6. `postal_name` — postal code + compatible name (token_set_ratio ≥ 0.5)
7. `rare_name_token` — rare business-name tokens (df ≤ 400)
8. `rare_addr_token` — rare address tokens (df ≤ 400)
9. `char_ngram` — name character 4-grams (shared ≥ 2)
10. `addr_ngram` — address character 4-grams (shared ≥ 2)
11. `token_overlap` — core token overlap (Jaccard-style)
12. `country_approx` — country-aware fuzzy name (token_set_ratio ≥ 0.6)

Every candidate keeps a provenance bitmask + retrieval ranks for downstream features.

### Streaming / Batched Processing
- Candidate generation in batches of 25k S1 entities (configurable)
- Hard cap of 1000 candidates per S1 entity
- DuckDB-backed indexes for disk-backed operations
- Explicit memory release between batches

### Features (91 total)
- **Name** (19): exact, fuzzy (ratio, Jaro-Winkler, token sort/set, partial), prefix/suffix, length, token coverage, n-gram Jaccard
- **Address** (17): exact, fuzzy, token coverage, length, numeric overlap, house number, postal, digit overlap, first/last token, completeness
- **Cross-field** (12): name×addr, min/max, high-high, high+number, high+postal, medium+high, presence flags, country equality/conflict
- **Rarity** (10): frequency logs, rare flags, IDF-weighted overlap
- **Blocking** (12): per-block retrieval flags, block count, retrieval ranks, source indicator
- **Multi-block Evidence** (4): exact block count, strong block count, weak block count, block bitmask

### Model & Training
- LightGBM binary classifier (pair-level)
- Entity-level 80/20 split (never pair-level)
- Hard-negative mining (4 tiers: very-hard, hard, medium, easy)
- Isotonic/Platt calibration selected by validation macro F0.5
- Threshold + ambiguity optimization directly on entity-level macro F0.5

## Memory-efficient by construction

* No Cartesian joins, no dense pairwise matrices.
* Twelve union blocks produce one merged candidate set per S1; every pair keeps a
  provenance bitmask plus retrieval ranks.
* Inverted indexes are `int32` numpy posting lists; tokens are tuples.
* Rare-token retrieval uses training-only frequency statistics (no test leakage).
* Candidate feature extraction memoises fuzzy bundles on the string pair.
* Entity-level 80/20 split; the split is by S1 entity, never by pair.
* **NEW**: Streaming candidate generation with configurable batch sizes
* **NEW**: DuckDB-backed disk indexes for large-scale deployment
* **NEW**: Block recall diagnostic to measure per-block recall/cost tradeoffs

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
│   ├── block_recall_diagnostic.csv   # NEW: per-block recall & cost
│   ├── candidate_recall.json
│   ├── validation_results.json
│   ├── error_analysis.csv
│   └── final_report.md
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