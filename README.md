# Business Entity Resolution — Amazon ML Challenge 2026

This repository implements scalable business entity resolution from Source 1 to
Source 2 and Source 3. Each Source-1 entity may have zero, one, or many matches.
The official metric is macro-averaged per-entity F0.5.

## Challenge-scale architecture

The production train/predict path is DuckDB-backed and batch-streamed.

- S2/S3 are materialized once into an on-disk DuckDB database.
- S1 is processed in bounded batches.
- Candidate generation happens inside DuckDB joins.
- Only the current candidate batch and its feature matrix are held in Python.
- Test predictions and candidate rows are written incrementally.
- No full-test candidate list or full-test feature matrix is accumulated.
- Final candidates are capped at 300 per S1 before ML scoring.

DuckDB supports larger-than-memory joins and can spill temporary work to disk;
the pipeline explicitly configures a temporary directory for this purpose.

## Blocking

The final candidate set is the exact candidate set fed to the matcher:

1. country + exact normalized name
2. country + exact heavy name
3. country + exact core name
4. exact normalized address
5. country + name + house number
6. country + postal + name similarity
7. country + name prefix + Jaro-Winkler
8. country + address prefix + Jaro-Winkler

Every candidate retains a block bitmask and block count. The merged set is capped
at 300 candidates per Source-1 entity before feature extraction.

## Features

The scalable matcher uses 25 frozen features covering:

- name exact/fuzzy similarity;
- address exact/fuzzy similarity;
- token overlap and length ratios;
- house-number and postal agreement;
- country equality/conflict;
- S2/S3 source;
- block provenance;
- cross-field agreement;
- field-presence signals.

## Model

LightGBM binary classifier:

- 96 leaves
- learning rate 0.05
- up to 2500 boosting rounds
- early stopping 150
- subsample 0.9
- column sampling 0.9
- L1 0.1
- L2 1.0
- deterministic seed 42

Training samples entities rather than pairs. Hard negatives are capped at 20 per
positive. The operating threshold is selected directly against validation
macro-F0.5.

No external business lookup, geocoding, registry lookup, commercial ER API, or
internet entity augmentation is used.

## Open-set country handling

Country is treated as a free-form string. The pipeline does not assume that only
US and India exist, so France and any other unseen test country remain eligible.

## Running

Install the pinned environment:

    pip install -r requirements.txt

Then run exactly one stage at a time:

    python run_pipeline.py profile --data-root student_resource
    python run_pipeline.py train --data-root student_resource
    python run_pipeline.py predict --data-root student_resource
    python run_pipeline.py validate-submission --data-root student_resource

Do not use the legacy all-in-memory training path for the challenge-scale run.

## Outputs

The predict stage writes:

    output/matching_results.tsv
    output/candidate_pairs.tsv

matching_results.tsv contains one row for every test Source-1 entity. An empty
second field means no match.

candidate_pairs.tsv is the exact final candidate set consumed by the ML matcher.
Every predicted match is therefore a member of its corresponding candidate list.

## Final package

The final archive must contain:

    output/
      matching_results.tsv
      candidate_pairs.tsv
    code/business_entity_resolution/
      src/
      README.md
      requirements.txt
    Documentation_template.md

Before packaging, run the supplied official validator:

    python3 utils/validate_submission.py \
      --matching output/matching_results.tsv \
      --candidate output/candidate_pairs.tsv \
      --test-dir dataset/test

The methodology document must be updated with the measured final validation,
candidate-recall, threshold, and runtime values before the final ZIP is submitted.

## Important score note

The code is designed to maximize the challenge's precision-heavy F0.5 objective,
but no implementation can honestly guarantee a hidden-test score such as 0.999
before the actual training and leaderboard evaluation. The repository therefore
records measured validation and candidate-recall results rather than promising a
specific leaderboard score.
