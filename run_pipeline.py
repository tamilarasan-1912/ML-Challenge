#!/usr/bin/env python3
"""Amazon ML Challenge 2026 - Business Entity Resolution CLI.

Usage
-----
    python run_pipeline.py profile             --data-root student_resource
    python run_pipeline.py train               --data-root student_resource
    python run_pipeline.py validate            --data-root student_resource
    python run_pipeline.py predict             --data-root student_resource
    python run_pipeline.py validate-submission --data-root student_resource
    python run_pipeline.py all                 --data-root student_resource

``--data-root`` is optional: with no value the dataset is auto-detected (env var
AMER_DATA_ROOT, cwd, Downloads, /workspace, ...).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# allow "python run_pipeline.py" from inside the project directory
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.pipeline import Pipeline, load_config  # noqa: E402
from src.utils import LOG, human_seconds, bytes_to_human as human_bytes  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Amazon ML Challenge 2026 - Business Entity Resolution",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "stage",
        choices=["profile", "train", "validate", "predict", "validate-submission", "report", "all"],
        help="pipeline stage to run",
    )
    p.add_argument("--data-root", default=None, help="path to student_resource (else auto-detect)")
    p.add_argument("--config", default=str(Path(__file__).resolve().parent / "config.yaml"))
    p.add_argument("--output-dir", default=None)
    p.add_argument("--reports-dir", default=None)
    p.add_argument("--models-dir", default=None)
    p.add_argument("--team-name", default=None, help="used when packaging the submission zip")
    p.add_argument("--zip", action="store_true", help="also build the final submission zip")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    if args.output_dir:
        cfg.output_dir = Path(args.output_dir)
    if args.reports_dir:
        cfg.reports_dir = Path(args.reports_dir)
    if args.models_dir:
        cfg.models_dir = Path(args.models_dir)

    t0 = time.perf_counter()
    pipeline = Pipeline(cfg, data_root=args.data_root)

    if args.stage == "profile":
        pipeline.profile()
    elif args.stage == "train":
        pipeline.train()
    elif args.stage == "validate":
        pipeline.validate()
    elif args.stage == "predict":
        pipeline.predict()
    elif args.stage == "validate-submission":
        pipeline.validate_submission()
    elif args.stage == "report":
        from src.reporting import generate_final_report

        p = generate_final_report(cfg.reports_dir, cfg.models_dir, cfg.output_dir)
        LOG.info("wrote %s", p)
    elif args.stage == "all":
        summary = pipeline.run_all()
        if args.zip:
            from build_zip import build_submission_zip

            team = args.team_name or cfg.section("submission").get("team_name", "team")
            path = build_submission_zip(
                pipeline.cfg, team_name=team, project_dir=Path(__file__).resolve().parent
            )
            LOG.info("submission zip: %s", path)
            summary["zip"] = str(path)
        _print_final_report(pipeline, summary)
        Path(cfg.reports_dir / "run_summary.json").write_text(
            json.dumps(summary, indent=2, default=str), encoding="utf-8"
        )

    LOG.info("stage '%s' finished in %s", args.stage, human_seconds(time.perf_counter() - t0))
    return 0


def _print_final_report(pipeline: "Pipeline", summary: dict) -> None:
    rep = pipeline.cfg.reports_dir
    bundle = {}
    vres = {}
    rec = {}
    try:
        bundle = json.loads((pipeline.cfg.models_dir / "artifacts_bundle.json").read_text())
    except Exception:
        pass
    try:
        vres = json.loads((rep / "validation_results.json").read_text())
    except Exception:
        pass
    try:
        rec = json.loads((rep / "candidate_recall.json").read_text())
    except Exception:
        pass

    def g(d, *keys, default="n/a"):
        for k in keys:
            if isinstance(d, dict) and k in d:
                d = d[k]
            else:
                return default
        return d

    # row counts from profile if present
    prof = {}
    try:
        prof = json.loads((rep / "data_profile.json").read_text())
    except Exception:
        pass
    counts = {}
    for split in ("train", "test"):
        for tag in ("S1", "S2", "S3"):
            counts[f"{split}_{tag}"] = g(prof, split, tag, "n_rows", default="n/a")

    ev = g(vres, "eval", default={})
    dec = g(vres, "decision", default={})
    print("\n" + "=" * 60)
    print("AMAZON ML CHALLENGE 2026 - FINAL REPORT")
    print("=" * 60)
    print(f"Dataset: {g(prof, 'data_root', default=str(pipeline.paths.root))}")
    for split in ("train", "test"):
        for tag in ("S1", "S2", "S3"):
            print(f"{split.capitalize()} Source{tag[1:]} rows: {counts[f'{split}_{tag}']}")
    print(f"Candidate Recall: {g(rec, 'candidate_recall')}")
    print(f"Candidate Reduction Ratio: {g(rec, 'reduction_ratio')}")
    print(f"Best Validation Macro F0.5: {g(ev, 'macro_f05')}")
    print(f"Precision: {g(ev, 'precision')}")
    print(f"Recall: {g(ev, 'recall')}")
    print(f"Singleton Performance: {g(ev, 'singleton_score')}")
    print(f"S2 Performance: {g(ev, 'by_source', 'S2', 'f05')}")
    print(f"S3 Performance: {g(ev, 'by_source', 'S3', 'f05')}")
    print(f"Best Global Threshold: {g(dec, 'threshold')}")
    print(f"Best S2 Threshold: {g(dec, 'threshold_s2')}")
    print(f"Best S3 Threshold: {g(dec, 'threshold_s3')}")
    print(f"Peak RAM: {human_bytes(g(summary, 'train', 'peak_mem', default=0))}")
    print(f"Training Runtime: {human_seconds(g(summary, 'train', 'runtime', default=0))}")
    print(f"Inference Runtime: {human_seconds(g(summary, 'predict', 'runtime', default=0))}")
    sub = g(summary, "submission", "official", default={})
    internal = g(summary, "submission", "internal_pass", default="n/a")
    overall = "PASS" if (sub.get("pass_") and internal) else ("FAIL" if sub.get("ran") else str(internal))
    print(f"Submission Validation: {overall}")
    print(f"Matching Output: {pipeline.cfg.output_dir / 'matching_results.tsv'}")
    print(f"Candidate Output: {pipeline.cfg.output_dir / 'candidate_pairs.tsv'}")
    print(f"Final ZIP: {g(summary, 'zip', default='(run with --zip to build)')}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
