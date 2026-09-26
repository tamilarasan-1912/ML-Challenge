# Documentation — Business Entity Resolution

> Team: <team>
> Model: LightGBM (MIT), feature-based, far below the 8B-parameter limit
> Metric: macro-averaged per-Source-1-entity F0.5

## 1. Methodology

The system resolves each Source-1 entity against Source 2 and Source 3. A Source-1
entity is allowed to have zero, one, or many matches. Decisions are made from a
pairwise LightGBM probability followed by an entity-level threshold, so singletons
are not forced to match.

The challenge-scale implementation is DuckDB-backed and batch-streamed: Source 2/3
records remain in a persistent on-disk DuckDB database; only a bounded Source-1
batch, its final candidates, and its feature matrix are held in Python memory. The
full test candidate set is never accumulated in a Python list or single dense
feature matrix.

No external business lookup, registry, geocoding service, ER API, or internet
entity augmentation is used.

## 2. Data loading and normalization

Input files are read explicitly as TSV. DuckDB materializes compact normalized
tables with light normalized name, heavy name, legal-token-stripped core name,
normalized address, free-form normalized country, name/address prefixes, and
house-number/postal-code candidates.

Country is open-set: the pipeline does not restrict values to US/India. Unseen
labels such as France remain valid strings and are processed normally.

## 3. Blocking / candidate generation

The final candidate set is the exact set passed to the pair matcher. Eight scalable
union blocks are used:

1. country + exact normalized name;
2. country + exact heavy name;
3. country + exact core name;
4. exact normalized address;
5. country + name + house number;
6. country + postal code + name similarity;
7. country + 3-character name prefix + Jaro-Winkler similarity;
8. country + 4-character address prefix + Jaro-Winkler similarity.

Each candidate stores a bitmask and block count. Per-block fuzzy retrieval is capped,
then the merged final candidate set is capped at 300 records per Source-1 entity.
This cap is applied before model scoring, so candidate_pairs.tsv is the exact
candidate set consumed by inference.

The candidate set is deliberately much smaller than a Cartesian S1 × (S2+S3)
comparison. DuckDB performs the large joins and can spill larger-than-memory
operations to disk; Python receives only the current batch.

## 4. Pairwise features

The scalable matcher currently uses 25 frozen features covering name similarity,
address similarity, house number, postal code, country compatibility, source,
blocking provenance, cross-field agreement, and field presence.

All feature computation is performed only for the final candidate set.

## 5. Model and training

LightGBM binary classification is used. Training samples Source-1 entities rather
than pairs, preserving entity independence between train and validation. Labelled
positive links are added to the sampled training candidate pool when a raw block
misses them; raw blocking recall is measured separately.

Negatives are capped at 20 per positive for the final training matrix. Validation
uses the complete candidate set for the held-out sampled entities.

Model configuration: 96 leaves, learning rate 0.05, up to 2500 boosting rounds,
early stopping 150 rounds, row/column subsampling 0.9, L1 0.1, L2 1.0, fixed seed 42.

## 6. Decision policy

The threshold is selected directly on the entity-level macro F0.5 metric. No entity
is forced to receive a match. The high-confidence floor is 0.999. Duplicate IDs
are removed before output.

## 7. Validation and measured results

The scalable engine writes reports/candidate_recall.json,
reports/validation_results.json, and models/artifacts_bundle.json after training.

The final document must be updated from those measured artifacts before the final
submission ZIP is produced. No score is fabricated in this template.

| Metric | Measured value |
|---|---|
| Raw candidate recall | <filled after challenge-scale run> |
| Mean candidates/S1 | <filled after challenge-scale run> |
| P95 / P99 / maximum | <filled after challenge-scale run> |
| Validation macro F0.5 | <filled after challenge-scale run> |
| Validation precision / recall | <filled after challenge-scale run> |
| Singleton score | <filled after challenge-scale run> |
| Selected threshold | <filled after challenge-scale run> |

## 8. Test inference and outputs

Test S1 is processed in fixed-size batches. For every batch the engine generates
the final candidates, computes features, scores with LightGBM, applies the selected
threshold, writes matching and candidate rows, and releases batch objects.

Therefore the complete test set is never represented as one in-memory candidate
list or feature matrix.

The required outputs are:
- output/matching_results.tsv
- output/candidate_pairs.tsv

Every test S1 is written exactly once. Empty lists are emitted as an empty second
TSV field. Matching IDs are restricted to S2/S3 records from the test candidate
tables.

## 9. Reproducibility and audit

Dependencies are pinned in requirements.txt. Run:

    python run_pipeline.py train --data-root student_resource
    python run_pipeline.py predict --data-root student_resource
    python run_pipeline.py validate-submission --data-root student_resource

The official validator supplied by the challenge remains the final format/ID
authority.

## 10. Fair play and model licence

All entity-resolution evidence comes only from the supplied challenge data.
LightGBM is MIT-licensed. The model is a gradient-boosted tree model and is far
below the challenge's 8-billion-parameter ceiling.
