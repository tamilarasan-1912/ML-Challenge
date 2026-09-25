# Documentation — Business Entity Resolution

> Team: `<team>`
> Model: LightGBM (MIT), feature-based, ≪ 8B parameters
> Metric optimised: macro-averaged per-S1-entity F0.5

Numbers marked **(measured)** are written automatically by `run_pipeline.py` into
`reports/`. Any cell left as `<fill after real run>` is intentionally blank until
the pipeline is executed against the real `student_resource` data; this project
never fabricates metrics.

---

## 1. Methodology

The task is entity resolution between a deduplicated reference source S1 and two
noisy sources S2/S3. Each S1 entity may have zero, one, or many true matches. The
official score is the macro-average of the per-entity F0.5, so the optimisation
target is precision-weighted and *singletons* (S1 entities with no match) count
fully.

The system follows the classic blocking → feature → classifier → clustering-free
decision architecture:

1. **Deterministic normalisation** into multiple text views.
2. **Multi-block candidate generation** (ten union blocks) producing one merged
   candidate set per S1 entity, with provenance recorded per pair.
3. **Pairwise feature extraction** across name, address, cross-field, rarity and
   blocking families.
4. **LightGBM** binary classifier trained on hard-negative-mined pairs.
5. **Probability calibration** (isotonic / Platt), retained only on validation gain.
6. **Entity-level decision layer** that optimises thresholds, per-source
   thresholds, ambiguity handling and singleton behaviour directly against
   validation macro F0.5.

No external lookup of any kind is used.

## 2. Preprocessing

Files are read with Polars `scan_csv`/`read_csv`, tab-separated, all columns forced
to string (`infer_schema_length=0`), ragged lines tolerated. `__MACOSX`, `.DS_Store`
and `._*` files are excluded during discovery. Columns are resolved tolerantly to
the canonical schema (`entity_id`, `business_name`, `business_address`, `country`).

## 3. Normalisation (and why multiple views are kept)

Text is never collapsed into a single string. For every record we retain the
original plus several derived views:

* **light** — Unicode NFKC, fancy punctuation folded, accents stripped, casefold,
  `&`/`+` → `and`, punctuation → spaces, whitespace collapsed;
* **heavy** — light plus legal-suffix and abbreviation canonicalisation
  (`private limited`→`pvt ltd`, `corporation`→`corp`, `company`→`co`, `limited`→`ltd`);
* **core** — heavy with legal/business-type tokens removed (the discriminative core);
* **tokens** — the token list of the light view.

Addresses add heavy normalisation (road→rd, street→st, avenue→ave, leading zeros
stripped), numeric tokens, house number, digit set and postal/PIN candidates
(US 5-digit, India 6-digit, US ZIP+4). Countries are canonicalised through an
open-set mapping and are never restricted to a fixed set — unseen values such as
France flow through as their own group.

## 4. Blocking and candidate generation

Ten blocks are unioned per S1 entity (bitmask `retrieved_by_*` plus
`number_of_blocks` recorded on every pair):

1. country + normalised name
2. country + core name
3. exact normalised address
4. name + house number
5. postal/PIN + compatible name
6. rare business-name token (df ≤ 400 blocks only)
7. rare address token
8. character 4-gram overlap (≥ 2 shared grams)
9. token-overlap retrieval
10. country-aware approximate retrieval (token-set ≥ 0.6)

Cost control: frequency caps on posting lists, per-block top-k caps, rare-token
prioritisation, `int32` posting arrays, and a merged per-entity candidate map.
No Cartesian join and no dense matrix is ever constructed.

### Measured candidate statistics (measured)

| Metric | Value |
|---|---|
| candidate recall | `<reports/candidate_recall.json: candidate_recall>` |
| reduction ratio | `<...reduction_ratio>` |
| mean / median candidates per S1 | `<...mean_candidates>` / `<...median_candidates>` |
| p95 / p99 / max | `<...p95_candidates>` / `<...p99_candidates>` / `<...max_candidates>` |
| S1→S2 candidate recall | `<...by_source.S2.recall>` |
| S1→S3 candidate recall | `<...by_source.S3.recall>` |

## 5. Feature engineering

71 features in five frozen, versioned families (order is fixed in
`src/features.py::FEATURE_NAMES`):

* **Name (19)** — exact/light/heavy/core equality, RapidFuzz ratio, Jaro-Winkler,
  partial ratio, token-sort, token-set, partial token-set, core token-set, prefix
  and suffix similarity, length/token-count deltas, common-token count, token and
  core-token coverage, n-gram Jaccard.
* **Address (17)** — exact/light/heavy equality, ratio, token-sort, token-set,
  coverage, length deltas, numeric-token overlap, house-number match, postal
  match, digit overlap, first/last token match, completeness agreement, missingness.
* **Cross-field (12)** — name×address, min/max, name-high-address-high,
  name-high-number-match, name-high-postal-match, name-medium-address-high,
  presence flags, country equal/conflict/present.
* **Rarity (9)** — log name frequency (S1 and candidate), rare-name flag, core-name
  and address frequency, rare-name-token and rare-address-token counts, IDF-weighted
  token overlap, token overlap.
* **Blocking (14)** — the ten `retrieved_by_*` flags, `number_of_blocks`,
  name and n-gram retrieval ranks, `cand_is_s3`.

Rarity statistics are computed on the training corpus only; for test inference they
are recomputed from the provided test data alone, so no label or test-set leakage
occurs.

## 6. Hard-negative mining

Negatives are stratified into very-hard / hard / medium / easy using the full
feature space. Very-hard and hard negatives are always kept; medium and easy are
sampled (0.35 / 0.05). The total negative budget is capped at 20× the number of
positives. Validation always uses the *complete* candidate set so measured F0.5
reflects deployment.

## 7. Model

LightGBM binary classifier, `num_leaves=96`, `learning_rate=0.05`,
`min_child_samples=20`, `subsample=0.9`, `colsample_bytree=0.9`,
`reg_alpha=0.1`, `reg_lambda=1.0`, early stopping (150 rounds) on the entity-level
validation fold, `deterministic=True`, `force_row_wise=True`, fixed seed.
Inputs are the 71 frozen features; the objective is `P(pair is a true match)`.

## 8. Calibration

Isotonic and Platt (sigmoid) calibrators are fitted on validation pairs. Each is
evaluated by the *final macro F0.5* at a fixed 0.5 threshold and the best is
retained; if neither helps, `none` is kept.

## 9. Threshold optimisation

A coarse grid (0.30 … 0.975) followed by a fine grid (±0.10, step 0.005) around the
best point, scoring macro F0.5 at every step. Source-specific thresholds for S2 and
S3 are then searched independently and kept only on validation improvement. No
threshold is ever assumed to be 0.5.

## 10. Singleton and ambiguity handling

* An S1 entity with no candidates predicts the empty set; an S1 entity whose best
  candidate is below the floor predicts the empty set. Nothing is forced.
* A `high_conf` override (≥ 0.995) guarantees near-certain pairs survive a high floor.
* Ambiguity knobs (`margin`, `top_only`, `max_matches_per_entity`) handle near-ties
  in the top scores. Each variant is scored on validation and kept only if it
  improves macro F0.5.

## 11. Optional S2↔S3 consistency

A conservative demotion rule: a marginal accepted match (p < 0.5) is dropped when a
much stronger sibling source dominates (ratio < 0.55). It only *removes* weak
matches and never adds new ones, so recursive error propagation is impossible. It
is retained only when the ablation ladder improves.

## 12. Validation protocol

The split is by **Source-1 entity** (80/20, seed 42) — never by pair. The
full candidate set of held-out entities is scored, and all thresholds/decision
knobs are chosen on that fold.

## 13. Experiments

`python run_experiments.py ablation` runs the 13-step ladder and writes
`reports/ablation_results.csv`; `variants` writes the five controlled submission
variants A–E to `reports/leaderboard_variants.csv`; `tune` runs a bounded
hyper-parameter search to `experiments/tuning_results.csv`.
`experiments/experiment_log.csv` is append-only and records every run.

## 14. Error analysis

`reports/error_analysis.csv` lists every false positive and false negative on the
validation fold with the S1/candidate ids, raw names and addresses, country,
probability, true label, blocking provenance, block count, and the top/second
probability and margin for that entity.

## 15. Final inference

For the test split: load → normalise → build indexes → generate the merged
candidate set → extract features → LightGBM → calibration → entity-level decision →
write `output/matching_results.tsv` and `output/candidate_pairs.tsv`. Every test S1
entity is represented exactly once; singletons carry an empty field.

## 16. Computational constraints

Designed for constrained RAM: streaming reads, `int32` posting lists, per-entity
candidate maps, memoised fuzzy bundles, no dense similarity matrices. Runtime and
peak RSS are reported by the pipeline.

## 17. Model licence

LightGBM is MIT-licensed; a feature-based gradient-boosted tree ensemble is orders
of magnitude below the 8B parameter limit. All dependencies are MIT / BSD / Apache-2.0
(see `requirements.txt`). No external data, lookup, geocoding, or registry is used.

## 18. Reproduction

```bash
pip install -r requirements.txt
python run_pipeline.py all --data-root student_resource --zip --team-name <team>
```

| Step | Result (measured) |
|---|---|
| Best validation macro F0.5 | `<reports/validation_results.json: eval.macro_f05>` |
| Precision / Recall | `<...eval.precision>` / `<...eval.recall>` |
| Singleton score | `<...eval.singleton_score>` |
| S2 / S3 F0.5 | `<...eval.by_source.S2.f05>` / `<...eval.by_source.S3.f05>` |
| Best global / S2 / S3 threshold | `<...decision.threshold>` / `<...decision.threshold_s2>` / `<...decision.threshold_s3>` |
| Official validator | `<reports/submission_validation.json: official.stdout>` |
