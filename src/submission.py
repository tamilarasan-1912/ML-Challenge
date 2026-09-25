"""Submission writing + internal structural validation.

Outputs
-------
output/matching_results.tsv : source1_entity_id <TAB> matched_entity_ids
output/candidate_pairs.tsv  : source1_entity_id <TAB> candidate_entity_ids

Structural rules enforced here (independent of the official validator):
  * exactly one row per test S1 entity, no duplicates, no omissions
  * matches only reference S2/S3 ids that exist in the test data
  * every accepted match appears in that entity's candidate list
  * no duplicate ids within a list, singletons keep an empty field
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import polars as pl

from .candidates import CandidatePair
from .indexing import CandidateSpace
from .io_utils import write_tsv
from .utils import LOG, ensure_dir


def write_matching_results(
    out_path: Path,
    s1_ids: Sequence[str],
    preds: Dict[int, Set[int]],
    cs: CandidateSpace,
) -> None:
    ensure_dir(Path(out_path).parent)
    rows = []
    for i, sid in enumerate(s1_ids):
        gids = preds.get(i, set())
        matched = ",".join(sorted(cs.entity_ids[g] for g in gids))
        rows.append((sid, matched))
    lines = ["source1_entity_id\tmatched_entity_ids"]
    lines += [f"{a}\t{b}" for a, b in rows]
    Path(out_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    LOG.info("wrote %s (%d rows)", out_path, len(rows))


def write_candidate_pairs(
    out_path: Path,
    s1_ids: Sequence[str],
    pairs: Sequence[CandidatePair],
    cs: CandidateSpace,
) -> None:
    ensure_dir(Path(out_path).parent)
    by_s1: Dict[int, List[str]] = {}
    for p in pairs:
        by_s1.setdefault(p.s1_index, []).append(cs.entity_ids[p.gid])
    lines = ["source1_entity_id\tcandidate_entity_ids"]
    for i, sid in enumerate(s1_ids):
        cands = sorted(set(by_s1.get(i, [])))
        lines.append(f"{sid}\t{','.join(cands)}")
    Path(out_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    LOG.info("wrote %s (%d rows)", out_path, len(s1_ids))


# --------------------------------------------------------------------------- #
# Internal structural checks
# --------------------------------------------------------------------------- #
def internal_checks(
    matching_path: Path,
    candidate_path: Path,
    test_s1_ids: Sequence[str],
    test_s2_ids: Sequence[str],
    test_s3_ids: Sequence[str],
) -> Tuple[bool, List[str]]:
    problems: List[str] = []
    s2 = set(test_s2_ids)
    s3 = set(test_s3_ids)
    s1 = set(test_s1_ids)

    def read(path: Path) -> Tuple[List[str], List[str]]:
        text = path.read_text(encoding="utf-8").splitlines()
        if not text:
            return [], []
        header = text[0].split("\t")
        rows = [ln.split("\t") for ln in text[1:] if ln != ""]
        return header, rows

    mh, mrows = read(matching_path)
    if mh != ["source1_entity_id", "matched_entity_ids"]:
        problems.append(f"matching_results header wrong: {mh}")
    if len(mrows) != len(test_s1_ids):
        problems.append(f"matching row count {len(mrows)} != test S1 count {len(test_s1_ids)}")
    seen = set()
    for row in mrows:
        if len(row) != 2:
            problems.append(f"matching row malformed: {row}")
            continue
        sid, matched = row
        if sid not in s1:
            problems.append(f"unknown S1 id in matching: {sid}")
        if sid in seen:
            problems.append(f"duplicate S1 row: {sid}")
        seen.add(sid)
        if matched.strip() == "":
            continue
        ids = [x for x in matched.split(",") if x]
        if len(ids) != len(set(ids)):
            problems.append(f"duplicate matched ids for {sid}: {matched}")
        for mid in ids:
            if mid.startswith("S1-"):
                problems.append(f"S1 id used as match for {sid}: {mid}")
            elif not (mid in s2 or mid in s3):
                problems.append(f"match id not in test: {mid} (for {sid})")
    missing = s1 - seen
    if missing:
        problems.append(f"{len(missing)} test S1 entities omitted from matching output")

    ch, crows = read(candidate_path)
    if ch != ["source1_entity_id", "candidate_entity_ids"]:
        problems.append(f"candidate_pairs header wrong: {ch}")
    cand_map: Dict[str, Set[str]] = {}
    cseen = set()
    for row in crows:
        if len(row) != 2:
            problems.append(f"candidate row malformed: {row}")
            continue
        sid, cands = row
        if sid in cseen:
            problems.append(f"duplicate S1 row in candidates: {sid}")
        cseen.add(sid)
        ids = [x for x in cands.split(",") if x]
        if len(ids) != len(set(ids)):
            problems.append(f"duplicate candidate ids for {sid}")
        for cid in ids:
            if not (cid in s2 or cid in s3):
                problems.append(f"candidate id not in test: {cid}")
        cand_map[sid] = set(ids)

    # every accepted match must appear in the candidate list
    for row in mrows:
        if len(row) != 2:
            continue
        sid, matched = row
        if matched.strip() == "":
            continue
        cands = cand_map.get(sid, set())
        for mid in matched.split(","):
            if mid and mid not in cands:
                problems.append(f"match {mid} for {sid} not in candidate list")

    ok = len(problems) == 0
    return ok, problems
