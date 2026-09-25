"""Optional S2 <-> S3 cross-source consistency.

When training ground truth supports it, we can build a symmetric constraint: if
S1--S2 and S1--S3 are both strong and S2--S3 are mutually consistent, accept both;
if a candidate is contradicted by a much stronger sibling source, demote it.

This is deliberately conservative: it only *removes* marginal matches (never adds
weak ones), so it cannot create recursive error propagation. The caller keeps it
only if validation macro F0.5 improves.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np

from .candidates import CandidatePair
from .evaluation import evaluate
from .indexing import CandidateSpace
from .utils import LOG


@dataclass
class ConsistencyConfig:
    enabled: bool = False
    demote_ratio: float = 0.55   # drop a match if p < ratio * best_match_prob_of_sibling_src
    min_prob: float = 0.5        # only demote matches that are already marginal


def apply_consistency(
    preds: Dict[int, Set[int]],
    pairs: Sequence[CandidatePair],
    probs: np.ndarray,
    n_s1: int,
    cs_order: Sequence[str],
    cs_source_of: np.ndarray,
    cfg: ConsistencyConfig,
) -> Dict[int, Set[int]]:
    if not cfg.enabled or cs_order is None or len(cs_order) < 2:
        return preds

    # index probability per (s1, gid)
    best_prob_by_src: Dict[Tuple[int, str], float] = {}
    prob_of: Dict[Tuple[int, int], float] = {}
    for r, p in enumerate(pairs):
        src = cs_order[int(cs_source_of[p.gid])]
        key = (p.s1_index, src)
        if prob_of.get((p.s1_index, p.gid), -1) < probs[r]:
            prob_of[(p.s1_index, p.gid)] = float(probs[r])
        if probs[r] > best_prob_by_src.get(key, -1):
            best_prob_by_src[key] = float(probs[r])

    out: Dict[int, Set[int]] = {}
    n_demoted = 0
    for i, matched in preds.items():
        if not matched:
            out[i] = set()
            continue
        # strongest accepted prob per source
        strongest = {}
        for g in matched:
            src = cs_order[int(cs_source_of[g])]
            pr = prob_of.get((i, g), best_prob_by_src.get((i, src), 0.0))
            if pr > strongest.get(src, -1):
                strongest[src] = pr
        overall = max(strongest.values()) if strongest else 0.0

        kept = set()
        for g in matched:
            src = cs_order[int(cs_source_of[g])]
            pr = prob_of.get((i, g), 0.0)
            if pr < cfg.min_prob and strongest.get(src, 0.0) < overall:
                # this source is clearly weaker than the dominating source
                if strongest.get(src, 0.0) < cfg.demote_ratio * overall:
                    n_demoted += 1
                    continue
            kept.add(g)
        out[i] = kept if kept else set()
    if n_demoted:
        LOG.info("consistency: demoted %d marginal matches", n_demoted)
    return out
