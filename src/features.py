"""Pairwise feature engineering.

Features are grouped into five families: name, address, cross-field, rarity and
blocking provenance. All string similarity work goes through RapidFuzz (C++
backed). Expensive fuzzy bundles are memoised on the (name_a, name_b) string pair,
which matters because business names repeat heavily in this domain.
"""
from __future__ import annotations

import math
from functools import lru_cache
from typing import Dict, List, Sequence, Tuple

import numpy as np
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

from .blocking import BLOCK_NAMES, BLOCKS, BlockingIndex
from .candidates import CandidatePair
from .indexing import CandidateSpace, FrequencyStats, SourceTable
from .utils import LOG

# Feature column order is FROZEN here; models and submission code rely on it.
FEATURE_NAMES: List[str] = [
    # ---- name ----
    "name_exact",
    "name_heavy_exact",
    "name_core_exact",
    "name_ratio",
    "name_jaro_winkler",
    "name_partial_ratio",
    "name_token_sort_ratio",
    "name_token_set_ratio",
    "name_partial_token_set_ratio",
    "name_core_token_set_ratio",
    "name_prefix_sim",
    "name_suffix_sim",
    "name_len_diff",
    "name_len_ratio",
    "name_tok_count_diff",
    "name_common_tokens",
    "name_token_coverage",
    "name_core_token_coverage",
    "name_ngram_jaccard",
    # ---- address ----
    "addr_exact",
    "addr_heavy_exact",
    "addr_ratio",
    "addr_token_sort_ratio",
    "addr_token_set_ratio",
    "addr_token_coverage",
    "addr_len_diff",
    "addr_len_ratio",
    "addr_num_overlap",
    "house_number_match",
    "postal_match",
    "addr_digit_overlap",
    "addr_first_token_match",
    "addr_last_token_match",
    "addr_completeness_agree",
    "addr_missing_s1",
    "addr_missing_cand",
    # ---- cross ----
    "name_x_addr",
    "name_addr_min",
    "name_addr_max",
    "name_high_addr_high",
    "name_high_number_match",
    "name_high_postal_match",
    "name_medium_addr_high",
    "name_present_both",
    "addr_present_both",
    "country_equal",
    "country_conflict",
    "country_present_both",
    # ---- rarity ----
    "name_freq_log_s1",
    "name_freq_log_cand",
    "name_is_rare",
    "core_name_freq_log_cand",
    "addr_freq_log_cand",
    "rare_name_token_count_s1",
    "rare_addr_token_count_s1",
    "idf_token_overlap",
    "token_overlap",
    # ---- blocking ----
    *[f"retrieved_by_{b}" for b in BLOCK_NAMES],
    "number_of_blocks",
    "name_retrieval_rank",
    "ngram_retrieval_rank",
    "cand_is_s3",
]

FEATURE_INDEX: Dict[str, int] = {n: i for i, n in enumerate(FEATURE_NAMES)}
N_FEATURES = len(FEATURE_NAMES)


# --------------------------------------------------------------------------- #
# Memoised fuzzy bundles
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=2_000_000)
def _name_fuzzy(a: str, b: str) -> Tuple[float, ...]:
    if not a or not b:
        return (0.0,) * 7
    return (
        fuzz.ratio(a, b) / 100.0,
        JaroWinkler.similarity(a, b),
        fuzz.partial_ratio(a, b) / 100.0,
        fuzz.token_sort_ratio(a, b) / 100.0,
        fuzz.token_set_ratio(a, b) / 100.0,
        fuzz.partial_token_set_ratio(a, b) / 100.0,
        fuzz.WRatio(a, b) / 100.0,
    )


@lru_cache(maxsize=2_000_000)
def _name_fuzzy_core(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return fuzz.token_set_ratio(a, b) / 100.0


@lru_cache(maxsize=2_000_000)
def _addr_fuzzy(a: str, b: str) -> Tuple[float, ...]:
    if not a or not b:
        return (0.0,) * 4
    return (
        fuzz.ratio(a, b) / 100.0,
        fuzz.token_sort_ratio(a, b) / 100.0,
        fuzz.token_set_ratio(a, b) / 100.0,
        fuzz.partial_ratio(a, b) / 100.0,
    )


def _prefix_sim(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    m = min(len(a), len(b))
    i = 0
    while i < m and a[i] == b[i]:
        i += 1
    return i / max(len(a), len(b))


def _suffix_sim(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    m = min(len(a), len(b))
    i = 0
    while i < m and a[-1 - i] == b[-1 - i]:
        i += 1
    return i / max(len(a), len(b))


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _coverage(small: set, big: set) -> float:
    if not small:
        return 0.0
    return len(small & big) / len(small)


# --------------------------------------------------------------------------- #
# Main extractor
# --------------------------------------------------------------------------- #
class FeatureExtractor:
    def __init__(
        self,
        s1: SourceTable,
        cs: CandidateSpace,
        stats: FrequencyStats,
        bi: BlockingIndex,
        rare_df: int = 400,
    ) -> None:
        self.s1 = s1
        self.cs = cs
        self.stats = stats
        self.bi = bi
        self.rare_df = rare_df
        # cache sets of digit/token sets per candidate to avoid recomputation
        self._cand_digits: Dict[int, set] = {}
        self._cand_tokset: Dict[int, set] = {}
        self._cand_core_tokset: Dict[int, set] = {}
        self._cand_nums: Dict[int, set] = {}

    # -- lazy per-candidate caches -- #
    def _digits(self, gid: int) -> set:
        d = self._cand_digits.get(gid)
        if d is None:
            d = set(ch for ch in self.cs.addr_light[gid] if ch.isdigit())
            self._cand_digits[gid] = d
        return d

    def _tokset(self, gid: int) -> set:
        t = self._cand_tokset.get(gid)
        if t is None:
            t = set(self.cs.name_tokens[gid])
            self._cand_tokset[gid] = t
        return t

    def _core_tokset(self, gid: int) -> set:
        t = self._cand_core_tokset.get(gid)
        if t is None:
            t = set(self.cs.name_core_tokens[gid])
            self._cand_core_tokset[gid] = t
        return t

    def _nums(self, gid: int) -> set:
        t = self._cand_nums.get(gid)
        if t is None:
            t = set(self.cs.addr_light[gid].split())
            t = {x for x in t if x.isdigit()}
            self._cand_nums[gid] = t
        return t

    # ------------------------------------------------------------------ #
    def extract(self, pairs: Sequence[CandidatePair], log_every: int = 500000) -> np.ndarray:
        n = len(pairs)
        X = np.zeros((n, N_FEATURES), dtype=np.float32)
        s1, cs, stats = self.s1, self.cs, self.stats
        rare_df = self.rare_df

        # cache s1 token/digit sets lazily
        s1_tokset: Dict[int, set] = {}
        s1_core_tokset: Dict[int, set] = {}
        s1_nums: Dict[int, set] = {}
        s1_digits: Dict[int, set] = {}

        def get_s1_sets(i: int):
            if i not in s1_tokset:
                s1_tokset[i] = set(s1.name_tokens[i])
                s1_core_tokset[i] = set(s1.name_core_tokens[i])
                s1_nums[i] = {x for x in s1.addr_light[i].split() if x.isdigit()}
                s1_digits[i] = set(ch for ch in s1.addr_light[i] if ch.isdigit())
            return s1_tokset[i], s1_core_tokset[i], s1_nums[i], s1_digits[i]

        for r, p in enumerate(pairs):
            i = p.s1_index
            g = p.gid
            row = X[r]

            # ---------------- name ----------------
            n1l = s1.name_light[i]
            n2l = cs.name_light[g]
            n1h = s1.name_heavy[i]
            n2h = cs.name_heavy[g]
            n1c = s1.name_core[i]
            n2c = cs.name_core[g]
            ratio, jw, pratio, tsr, tset, ptset, wratio = _name_fuzzy(n1l, n2l)
            core_tset = _name_fuzzy_core(n1c, n2c)
            toks1, ctoks1, nums1, dig1 = get_s1_sets(i)
            toks2 = self._tokset(g)
            ctoks2 = self._core_tokset(g)
            nums2 = self._nums(g)

            row[0] = 1.0 if n1l and n1l == n2l else 0.0
            row[1] = 1.0 if n1h and n1h == n2h else 0.0
            row[2] = 1.0 if n1c and n1c == n2c else 0.0
            row[3] = ratio
            row[4] = jw
            row[5] = pratio
            row[6] = tsr
            row[7] = tset
            row[8] = ptset
            row[9] = core_tset
            row[10] = _prefix_sim(n1l, n2l)
            row[11] = _suffix_sim(n1l, n2l)
            row[12] = abs(len(n1l) - len(n2l))
            row[13] = (min(len(n1l), len(n2l)) / max(len(n1l), len(n2l))) if (n1l and n2l) else 0.0
            row[14] = abs(len(toks1) - len(toks2))
            row[15] = len(toks1 & toks2)
            row[16] = _coverage(toks1, toks2)
            row[17] = _coverage(ctoks1, ctoks2)
            row[18] = p.ngram_jaccard

            # ---------------- address ----------------
            a1l = s1.addr_light[i]
            a2l = cs.addr_light[g]
            a1h = s1.addr_heavy[i]
            a2h = cs.addr_heavy[g]
            aratio, atsr, atset, apratio = _addr_fuzzy(a1l, a2l)
            atoks1 = set(s1.addr_tokens[i])
            atoks2 = set(cs.addr_tokens[g])
            row[19] = 1.0 if a1l and a1l == a2l else 0.0
            row[20] = 1.0 if a1h and a1h == a2h else 0.0
            row[21] = aratio
            row[22] = atsr
            row[23] = atset
            row[24] = _coverage(atoks1, atoks2)
            row[25] = abs(len(a1l) - len(a2l))
            row[26] = (min(len(a1l), len(a2l)) / max(len(a1l), len(a2l))) if (a1l and a2l) else 0.0
            row[27] = _jaccard(nums1, nums2)
            h1 = s1.house_no[i]
            h2 = cs.house_no[g]
            row[28] = 1.0 if (h1 and h1 == h2) else 0.0
            p1 = set(s1.postals[i])
            p2 = set(cs.postals[g])
            row[29] = 1.0 if (p1 & p2) else 0.0
            d2 = self._digits(g)
            row[30] = _jaccard(dig1, d2)
            ft1 = s1.addr_tokens[i][0] if s1.addr_tokens[i] else ""
            ft2 = cs.addr_tokens[g][0] if cs.addr_tokens[g] else ""
            row[31] = 1.0 if (ft1 and ft1 == ft2) else 0.0
            lt1 = s1.addr_tokens[i][-1] if s1.addr_tokens[i] else ""
            lt2 = cs.addr_tokens[g][-1] if cs.addr_tokens[g] else ""
            row[32] = 1.0 if (lt1 and lt1 == lt2) else 0.0
            row[33] = 1.0 if ((not a1l) == (not a2l)) else 0.0
            row[34] = 1.0 if not a1l else 0.0
            row[35] = 1.0 if not a2l else 0.0

            # ---------------- cross ----------------
            row[36] = ratio * aratio
            row[37] = min(ratio, aratio)
            row[38] = max(ratio, aratio)
            row[39] = 1.0 if (ratio >= 0.85 and aratio >= 0.85) else 0.0
            row[40] = 1.0 if (ratio >= 0.80 and row[28] > 0) else 0.0
            row[41] = 1.0 if (ratio >= 0.80 and row[29] > 0) else 0.0
            row[42] = 1.0 if (0.55 <= ratio < 0.85 and aratio >= 0.85) else 0.0
            row[43] = 1.0 if (n1l and n2l) else 0.0
            row[44] = 1.0 if (a1l and a2l) else 0.0
            c1 = s1.country_norm[i]
            c2 = cs.country_norm[g]
            row[45] = 1.0 if (c1 and c1 == c2) else 0.0
            row[46] = 1.0 if (c1 and c2 and c1 != c2) else 0.0
            row[47] = 1.0 if (c1 and c2) else 0.0

            # ---------------- rarity ----------------
            f1 = stats.name_freq.get(n1l, 0)
            f2 = stats.name_freq.get(n2l, 0)
            row[48] = math.log1p(f1)
            row[49] = math.log1p(f2)
            row[50] = 1.0 if (0 < f2 <= rare_df) else 0.0
            row[51] = math.log1p(stats.core_name_freq.get(n2c, 0))
            row[52] = math.log1p(stats.addr_freq.get(a2l, 0))
            row[53] = float(sum(1 for t in ctoks1 if 0 < stats.name_token_freq.get(t, 0) <= rare_df))
            row[54] = float(
                sum(
                    1
                    for t in atoks1
                    if not t.isdigit() and 0 < stats.addr_token_freq.get(t, 0) <= rare_df
                )
            )
            row[55] = p.idf_overlap
            row[56] = p.token_overlap

            # ---------------- blocking ----------------
            base = 57
            for k, bname in enumerate(BLOCK_NAMES):
                row[base + k] = 1.0 if (p.blocks & BLOCKS[bname]) else 0.0
            o = base + len(BLOCK_NAMES)
            row[o] = p.n_blocks
            row[o + 1] = float(min(p.name_retrieval_rank, 1000))
            row[o + 2] = float(min(p.ngram_retrieval_rank, 1000))
            row[o + 3] = 1.0 if int(cs.source_of[g]) == 1 else 0.0

            if log_every and (r + 1) % log_every == 0:
                LOG.info("  features: %d/%d pairs", r + 1, n)
        return X


def labels_for_pairs(pairs: Sequence[CandidatePair], positives: Dict[int, set]) -> np.ndarray:
    y = np.zeros(len(pairs), dtype=np.int8)
    for r, p in enumerate(pairs):
        if p.s1_index in positives and p.gid in positives[p.s1_index]:
            y[r] = 1
    return y
