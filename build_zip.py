#!/usr/bin/env python3
"""Package the final submission zip.

Structure produced:

    <team>_submission.zip
    |-- output/{matching_results.tsv, candidate_pairs.tsv}
    |-- code/business_entity_resolution/{src, README.md, requirements.txt, config.yaml, run_pipeline.py}
    |-- Documentation_template.md

No large intermediate artefacts (models, candidate feature matrices, synthetic
dev data) are included.
"""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path
from typing import Optional

CODE_FILES = [
    "README.md",
    "requirements.txt",
    "config.yaml",
    "run_pipeline.py",
    "run_experiments.py",
    "build_zip.py",
]
CODE_DIRS = ["src"]


def build_submission_zip(cfg, team_name: str, project_dir: Path) -> Path:
    out_zip = project_dir / f"{team_name}_submission.zip"
    match = cfg.output_dir / "matching_results.tsv"
    cand = cfg.output_dir / "candidate_pairs.tsv"
    if not match.exists() or not cand.exists():
        raise FileNotFoundError(
            f"Missing outputs: {match} / {cand}. Run `predict` first."
        )

    doc = project_dir / "Documentation_template.md"

    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(match, "output/matching_results.tsv")
        z.write(cand, "output/candidate_pairs.tsv")
        for f in CODE_FILES:
            p = project_dir / f
            if p.exists():
                z.write(p, f"code/business_entity_resolution/{f}")
        for d in CODE_DIRS:
            base = project_dir / d
            if not base.exists():
                continue
            for p in sorted(base.rglob("*.py")):
                if "__pycache__" in p.parts:
                    continue
                z.write(p, f"code/business_entity_resolution/{p.relative_to(project_dir)}")
        if doc.exists():
            z.write(doc, "Documentation_template.md")
        # include the report (small, useful for graders)
        rep = project_dir / "reports" / "final_report.md"
        if rep.exists():
            z.write(rep, "code/business_entity_resolution/reports/final_report.md")
    return out_zip


def main() -> int:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from src.pipeline import load_config

    ap = argparse.ArgumentParser()
    ap.add_argument("--team-name", required=True)
    ap.add_argument("--config", default=str(Path(__file__).resolve().parent / "config.yaml"))
    ap.add_argument("--project-dir", default=str(Path(__file__).resolve().parent))
    a = ap.parse_args()
    cfg = load_config(a.config)
    p = build_submission_zip(cfg, a.team_name, Path(a.project_dir))
    print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
