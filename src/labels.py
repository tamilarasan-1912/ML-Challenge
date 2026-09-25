"""Ground-truth label handling.

The task allows zero, one or many matches per Source-1 entity, so labels are
stored as ``s1_index -> set(global_candidate_id)``. Singletons (empty labels) are
first-class and must be learnable.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from .indexing import CandidateSpace, SourceTable
from .io_utils import parse_match_list
from .utils import LOG


@dataclass
class LabelSet:
    positives: Dict[int, Set[int]]        # s1_index -> set(gid)
    s1_with_matches: int
    s1_singletons: int
    n_positive_pairs: int
    unmatched_gt_ids: List[str]           # gt matches not present in the candidate space

    def match_count_distribution(self) -> Dict[int, int]:
        dist: Dict[int, int] = {}
        for s1i, gids in self.positives.items():
            k = len(gids)
            dist[k] = dist.get(k, 0) + 1
        return dist


def build_labels(
    gt_df,
    s1: SourceTable,
    cs: CandidateSpace,
) -> LabelSet:
    """Map ground-truth rows onto (s1_index, gid) positives."""
    s1_ids = gt_df["source1_entity_id"].cast(str).to_list()
    raw_matches = gt_df["matched_entity_ids"].to_list()

    s1_index = s1.id_to_idx()
    gid_of = {e: g for g, e in enumerate(cs.entity_ids)}

    positives: Dict[int, Set[int]] = {}
    unmatched: List[str] = []
    n_pairs = 0
    for sid, raw in zip(s1_ids, raw_matches):
        si = s1_index.get(sid)
        if si is None:
            # GT references an S1 id not present in the S1 file (should not happen)
            continue
        gids: Set[int] = set()
        for mid in parse_match_list(raw):
            g = gid_of.get(mid)
            if g is None:
                unmatched.append(mid)
                continue
            gids.add(g)
        positives[si] = gids
        n_pairs += len(gids)

    n_with = sum(1 for v in positives.values() if v)
    ls = LabelSet(
        positives=positives,
        s1_with_matches=n_with,
        s1_singletons=s1.n - n_with,
        n_positive_pairs=n_pairs,
        unmatched_gt_ids=unmatched,
    )
    LOG.info(
        "labels: %d S1 with matches, %d singletons, %d positive pairs, %d unmatched gt ids",
        ls.s1_with_matches,
        ls.s1_singletons,
        ls.n_positive_pairs,
        len(unmatched),
    )
    return ls
