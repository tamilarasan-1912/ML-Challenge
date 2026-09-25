# Final Report — Business Entity Resolution (Amazon ML Challenge 2026)

All figures below are measured from the supplied dataset by this pipeline. No external data, lookup, geocoding, or registry was used.

## Dataset statistics

- data root: `/workspace/project/ML-Challenge/dev/synthetic/student_resource`
- train S1: 1200 rows (blank names: 0)
- train S2: 1859 rows (blank names: 0)
- train S3: 651 rows (blank names: 0)
- test S1: 400 rows (blank names: 0)
- test S2: 657 rows (blank names: 0)
- test S3: 218 rows (blank names: 0)
- ground-truth rows: 1200; S1 with matches: 699; singletons: 501; multi-match entities: 375
- match-count distribution: {'0': 501, '1': 324, '2': 235, '3': 77, '4': 42, '5': 19, '6': 2}
- positives by source: {'S3': 651, 'S2': 649}

## Candidate generation

- candidate recall: 1.0
- candidate reduction ratio: 0.7899926958831341
- mean / median candidates per S1: 527.1183333333333 / 518.0
- p95 / p99 / max candidates: 662.05 / 707.01 / 721.0
- positive pairs: 1300 of 1300 retained
- S1->S2 candidate recall: 1.0 (649/649)
- S1->S3 candidate recall: 1.0 (651/651)
- country-specific candidate recall:
    - united states: 1.0 (714/714)
    - india: 1.0 (586/586)

## Validation performance

- **best validation macro F0.5: 0.9998611111111111**
- micro precision: 0.9963369963369964
- micro recall: 1.0
- singleton macro F0.5: 1.0
- match-entity macro F0.5: 0.9988179669030733
- tp / fp / fn: 272 / 1 / 0
- predicted pairs: 273
- evaluated entities: 1200 (with ground truth: 141)
- S1->S2: F0.5=0.9944367176634215, P=0.9930555555555556, R=1.0, tp/fp/fn=143/1/0
- S1->S3: F0.5=1.0, P=1.0, R=1.0, tp/fp/fn=129/0/0
- by country (macro F0.5):
    - united states: 0.9997427983539096 (n=648)
    - india: 1.0 (n=552)

## Decision logic

- global threshold: 0.3
- S2 threshold: None
- S3 threshold: None
- high-confidence override: 0.995
- ambiguity margin: 0.0; top-only: False; max matches/entity: 0
- chosen calibrator: isotonic

## Hard-negative mining

- total training candidate pairs: n/a
- positives: n/a; negatives: n/a
- tier counts (very-hard/hard/medium/easy): n/a / n/a / n/a / n/a
- selected for training: n/a (negatives: n/a)

## Ablation results

| Experiment | Macro F0.5 | Details |
|---|---|---|
| 1. exact matching only | 0.204108 | rule-based, no model |
| 2. + normalization (core name) | 0.092688 | rule-based |
| 3. + fuzzy name features | 0.937526 |  |
| 4. + fuzzy address features | 0.8825 |  |
| 5. + numeric/address features | 0.999491 |  |
| 6. + multi-blocking features | 0.8825 |  |
| 7. + rarity features | 0.999491 |  |
| 8. all features + hard negatives | 0.999491 |  |
| 9. + calibration + threshold opt | 0.999861 |  |
| 10. + tuned global threshold | 0.999861 | thr=0.3000 |
| 11. + source-specific thresholds | 0.999861 | s2=None s3=None |
| 12. + ambiguity logic | 0.999861 | margin=0.0 top_only=False max=0 |
| 13. + S2/S3 consistency (optional) | 0.999861 | only kept if > experiment 12 |

## Experiment log

| experiment_id | macro_f05 | precision | recall | candidate_recall | threshold | runtime | peak_memory | notes |
|---|---|---|---|---|---|---|---|---|
| train_1790350170 | 0.999861 | 0.996337 | 1.0 | 1.0 | 0.3 | 33.08 | 1315500032 | calib=isotonic;mining_sel=21588 |
| train_1790351081 | 0.999861 | 0.996337 | 1.0 | 1.0 | 0.3 | 33.17 | 1332047872 | calib=isotonic;mining_sel=21588 |
| train_1790351284 | 0.999861 | 0.996337 | 1.0 | 1.0 | 0.3 | 33.14 | 1332428800 | calib=isotonic;mining_sel=21588 |
| train_1790351482 | 0.999861 | 0.996337 | 1.0 | 1.0 | 0.3 | 33.32 | 1327566848 | calib=isotonic;mining_sel=21588 |

## Submission validation

- official validator ran: True
- official validator return code: 0
- official validator stdout: `PASS
matching rows: 400; test S1: 400; S2: 657; S3: 218`
- internal structural checks: PASS
- matching output: `output/matching_results.tsv`
- candidate output: `output/candidate_pairs.tsv`

## Failure patterns to watch

- False merges between common-name businesses sharing a city/PIN: mitigated by rarity features and the precision-heavy threshold.
- Missed matches where the address is absent on one side but the name is strong: mitigated by name-high-name-number and name-high-postal cross features.
- Singletons wrongly given a match: the dominant error mode; directly optimised by the threshold and ambiguity searches.
- Unseen-country test entities (e.g. France): handled open-set, never hard-coded.

## Compliance

- No external entity lookup, geocoding, registry, or commercial API was used.
- Model: LightGBM (MIT), feature-based, far below the 8B parameter cap.
- Dependencies: MIT / BSD / Apache-2.0 only (see requirements.txt).
- Every test S1 entity is represented exactly once; singletons keep an empty field.
