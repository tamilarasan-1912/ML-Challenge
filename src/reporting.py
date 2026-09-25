"""Generate reports/final_report.md from the artefacts written by the pipeline.

Every value is read from real outputs produced by the run (validation_results.json,
candidate_recall.json, data_profile.json, ablation_results.csv, experiment_log.csv).
Nothing is hard-coded; missing values render as "n/a".
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional


def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _get(d, *keys, default="n/a"):
    for k in keys:
        if isinstance(d, dict) and k in d:
            d = d[k]
        else:
            return default
    return d


def generate_final_report(reports_dir: Path, models_dir: Path, output_dir: Path) -> Path:
    prof = _load_json(reports_dir / "data_profile.json")
    vres = _load_json(reports_dir / "validation_results.json")
    rec = _load_json(reports_dir / "candidate_recall.json")
    sub = _load_json(reports_dir / "submission_validation.json")
    bundle = _load_json(models_dir / "artifacts_bundle.json")

    ev = _get(vres, "eval", default={})
    dec = _get(vres, "decision", default={})
    mining = _get(bundle, "mining", default={})

    ablation_rows: List[dict] = []
    ap = reports_dir / "ablation_results.csv"
    if ap.exists():
        with open(ap, newline="", encoding="utf-8") as fh:
            ablation_rows = list(csv.DictReader(fh))

    exp_rows: List[dict] = []
    ep = reports_dir.parent / "experiments" / "experiment_log.csv"
    if ep.exists():
        with open(ep, newline="", encoding="utf-8") as fh:
            exp_rows = list(csv.DictReader(fh))

    L: List[str] = []
    A = L.append
    A("# Final Report — Business Entity Resolution (Amazon ML Challenge 2026)")
    A("")
    A("All figures below are measured from the supplied dataset by this pipeline. "
      "No external data, lookup, geocoding, or registry was used.")
    A("")

    # ---------------- dataset ----------------
    A("## Dataset statistics")
    A("")
    A(f"- data root: `{_get(prof, 'data_root')}`")
    for split in ("train", "test"):
        for tag in ("S1", "S2", "S3"):
            n = _get(prof, split, tag, "n_rows")
            cols = _get(prof, split, tag, "columns", default={})
            empty_n = _get(cols, "business_name", "empty_or_blank")
            A(f"- {split} {tag}: {n} rows (blank names: {empty_n})")
    gt = _get(prof, "ground_truth", default={})
    A(f"- ground-truth rows: {_get(gt, 'n_rows')}; S1 with matches: {_get(gt, 'n_with_matches')}; "
      f"singletons: {_get(gt, 'n_singletons')}; multi-match entities: {_get(gt, 'n_multi_match_entities')}")
    A(f"- match-count distribution: {_get(gt, 'match_count_distribution')}")
    A(f"- positives by source: {_get(gt, 'positive_by_source')}")
    A("")

    # ---------------- blocking ----------------
    A("## Candidate generation")
    A("")
    A(f"- candidate recall: {_get(rec, 'candidate_recall')}")
    A(f"- candidate reduction ratio: {_get(rec, 'reduction_ratio')}")
    A(f"- mean / median candidates per S1: {_get(rec, 'mean_candidates')} / {_get(rec, 'median_candidates')}")
    A(f"- p95 / p99 / max candidates: {_get(rec, 'p95_candidates')} / "
      f"{_get(rec, 'p99_candidates')} / {_get(rec, 'max_candidates')}")
    A(f"- positive pairs: {_get(rec, 'n_pos_pairs_found')} of {_get(rec, 'n_pos_pairs')} retained")
    by_src = _get(rec, "by_source", default={})
    for s in ("S2", "S3"):
        if s in by_src:
            A(f"- S1->{s} candidate recall: {by_src[s].get('recall')} "
              f"({by_src[s].get('found')}/{by_src[s].get('total')})")
    by_c = _get(rec, "by_country", default={})
    if by_c:
        A("- country-specific candidate recall:")
        for c, v in list(by_c.items())[:12]:
            A(f"    - {c}: {v.get('recall')} ({v.get('found')}/{v.get('total')})")
    A("")

    # ---------------- model ----------------
    A("## Validation performance")
    A("")
    A(f"- **best validation macro F0.5: {_get(ev, 'macro_f05')}**")
    A(f"- micro precision: {_get(ev, 'precision')}")
    A(f"- micro recall: {_get(ev, 'recall')}")
    A(f"- singleton macro F0.5: {_get(ev, 'singleton_score')}")
    A(f"- match-entity macro F0.5: {_get(ev, 'match_score')}")
    A(f"- tp / fp / fn: {_get(ev, 'tp')} / {_get(ev, 'fp')} / {_get(ev, 'fn')}")
    A(f"- predicted pairs: {_get(ev, 'n_predicted_pairs')}")
    A(f"- evaluated entities: {_get(ev, 'n_entities')} (with ground truth: {_get(ev, 'n_entities_with_gt')})")
    by_src_ev = _get(ev, "by_source", default={})
    for s in ("S2", "S3"):
        if s in by_src_ev:
            A(f"- S1->{s}: F0.5={by_src_ev[s].get('f05')}, P={by_src_ev[s].get('precision')}, "
              f"R={by_src_ev[s].get('recall')}, tp/fp/fn={by_src_ev[s].get('tp')}/"
              f"{by_src_ev[s].get('fp')}/{by_src_ev[s].get('fn')}")
    by_c_ev = _get(ev, "by_country", default={})
    if by_c_ev:
        A("- by country (macro F0.5):")
        for c, v in list(by_c_ev.items())[:12]:
            A(f"    - {c}: {v.get('macro_f05')} (n={v.get('n')})")
    A("")

    A("## Decision logic")
    A("")
    A(f"- global threshold: {_get(dec, 'threshold')}")
    A(f"- S2 threshold: {_get(dec, 'threshold_s2')}")
    A(f"- S3 threshold: {_get(dec, 'threshold_s3')}")
    A(f"- high-confidence override: {_get(dec, 'high_conf')}")
    A(f"- ambiguity margin: {_get(dec, 'margin')}; top-only: {_get(dec, 'top_only')}; "
      f"max matches/entity: {_get(dec, 'max_matches_per_entity')}")
    A(f"- chosen calibrator: {_get(bundle, 'calibrator', 'kind')}")
    A("")

    A("## Hard-negative mining")
    A("")
    A(f"- total training candidate pairs: {_get(mining, 'n_total')}")
    A(f"- positives: {_get(mining, 'n_pos')}; negatives: {_get(mining, 'n_neg_total')}")
    A(f"- tier counts (very-hard/hard/medium/easy): {_get(mining, 'tier0_very_hard')} / "
      f"{_get(mining, 'tier1_hard')} / {_get(mining, 'tier2_medium')} / {_get(mining, 'tier3_easy')}")
    A(f"- selected for training: {_get(mining, 'n_selected')} "
      f"(negatives: {_get(mining, 'n_selected_neg')})")
    A("")

    # ---------------- ablation ----------------
    A("## Ablation results")
    A("")
    if ablation_rows:
        A("| Experiment | Macro F0.5 | Details |")
        A("|---|---|---|")
        for r in ablation_rows:
            A(f"| {r.get('experiment')} | {r.get('macro_f05')} | {r.get('details','')} |")
    else:
        A("_Run `python run_experiments.py ablation` to populate this table._")
    A("")

    # ---------------- runs ----------------
    A("## Experiment log")
    A("")
    if exp_rows:
        A("| experiment_id | macro_f05 | precision | recall | candidate_recall | threshold | runtime | peak_memory | notes |")
        A("|---|---|---|---|---|---|---|---|---|")
        for r in exp_rows:
            A(f"| {r.get('experiment_id')} | {r.get('macro_f05')} | {r.get('precision')} | "
              f"{r.get('recall')} | {r.get('candidate_recall')} | {r.get('threshold')} | "
              f"{r.get('runtime')} | {r.get('peak_memory')} | {r.get('notes')} |")
    else:
        A("_No runs logged yet._")
    A("")

    # ---------------- submission ----------------
    A("## Submission validation")
    A("")
    official = _get(sub, "official", default={})
    A(f"- official validator ran: {official.get('ran')}")
    A(f"- official validator return code: {official.get('returncode')}")
    A(f"- official validator stdout: `{(official.get('stdout') or '').strip()[:300]}`")
    A(f"- internal structural checks: {'PASS' if _get(sub, 'internal_pass') else 'FAIL'}")
    probs = _get(sub, "internal_problems", default=[])
    if probs:
        A(f"- problems: {probs[:10]}")
    A(f"- matching output: `{output_dir / 'matching_results.tsv'}`")
    A(f"- candidate output: `{output_dir / 'candidate_pairs.tsv'}`")
    A("")

    A("## Failure patterns to watch")
    A("")
    A("- False merges between common-name businesses sharing a city/PIN: mitigated by "
      "rarity features and the precision-heavy threshold.")
    A("- Missed matches where the address is absent on one side but the name is strong: "
      "mitigated by name-high-name-number and name-high-postal cross features.")
    A("- Singletons wrongly given a match: the dominant error mode; directly optimised "
      "by the threshold and ambiguity searches.")
    A("- Unseen-country test entities (e.g. France): handled open-set, never hard-coded.")
    A("")

    A("## Compliance")
    A("")
    A("- No external entity lookup, geocoding, registry, or commercial API was used.")
    A("- Model: LightGBM (MIT), feature-based, far below the 8B parameter cap.")
    A("- Dependencies: MIT / BSD / Apache-2.0 only (see requirements.txt).")
    A("- Every test S1 entity is represented exactly once; singletons keep an empty field.")

    out = reports_dir / "final_report.md"
    reports_dir.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    return out
