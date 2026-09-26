#!/usr/bin/env python3
"""Stand-in official validator (same CLI contract as the challenge one)."""
import argparse, sys
from pathlib import Path

def read_pairs(p):
    rows = {}
    for ln in Path(p).read_text(encoding="utf-8").splitlines()[1:]:
        if not ln:
            continue
        parts = ln.split("\t")
        if len(parts) != 2:
            rows.setdefault("__malformed__", 0)
            rows["__malformed__"] += 1
            continue
        rows[parts[0]] = parts[1]
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matching", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--test-dir", required=True)
    a = ap.parse_args()
    test_dir = Path(a.test_dir)
    s1 = Path(test_dir) / "test_source1.tsv"
    s2 = Path(test_dir) / "test_source2.tsv"
    s3 = Path(test_dir) / "test_source3.tsv"
    ids = lambda p: {ln.split("\t")[0] for ln in p.read_text(encoding="utf-8").splitlines()[1:] if ln}
    s1_ids, s2_ids, s3_ids = ids(s1), ids(s2), ids(s3)
    match = read_pairs(a.matching)
    cand = read_pairs(a.candidate)
    errs = []
    if len(match) != len(s1_ids):
        errs.append(f"matching rows {len(match)} != S1 {len(s1_ids)}")
    cand_sets = {k: set(v.split(",")) - {""} for k, v in cand.items()}
    for sid, matched in match.items():
        if sid not in s1_ids:
            errs.append(f"unknown S1 {sid}")
        for mid in (set(matched.split(",")) - {""}):
            if not (mid in s2_ids or mid in s3_ids):
                errs.append(f"match {mid} not in test")
            if mid not in cand_sets.get(sid, set()):
                errs.append(f"match {mid} not in candidates of {sid}")
    missing = s1_ids - set(match)
    if missing:
        errs.append(f"{len(missing)} S1 entities missing")
    if errs:
        print("FAIL")
        for e in errs[:20]:
            print(" ", e)
        sys.exit(1)
    print("PASS")
    print(f"matching rows: {len(match)}; test S1: {len(s1_ids)}; S2: {len(s2_ids)}; S3: {len(s3_ids)}")
    sys.exit(0)

if __name__ == "__main__":
    main()
