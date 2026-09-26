"""Blocking: build the indexes that drive multi-block candidate generation.

The union of the blocks below must maximise *candidate recall*. Precision is
recovered later by the learned matcher, so blocking is intentionally permissive
while still bounded in cost (frequency caps, rare-token prioritisation).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import indexing as idx
from .indexing import CandidateSpace, FrequencyStats, SourceTable
from .utils import LOG


# --------------------------------------------------------------------------- #
# Block bit assignments (kept explicit so features are self-documenting)
# --------------------------------------------------------------------------- #
BLOCKS: Dict[str, int] = {
    "exact_name": 1 << 0,        # 1  country + normalized name
    "core_name": 1 << 1,         # 2  country + core name
    "exact_address": 1 << 2,     # 3  exact normalized address
    "name_housenumber": 1 << 3,  # 4  name + house number
    "postal_name": 1 << 4,       # 5  postal + compatible name
    "rare_name_token": 1 << 5,   # 6  rare business-name token
    "rare_addr_token": 1 << 6,   # 7  rare address token
    "char_ngram": 1 << 7,        # 8  character n-gram retrieval
    "token_overlap": 1 << 8,     # 9  token-overlap retrieval
    "country_approx": 1 << 9,    # 10 country-aware approximate retrieval
    "addr_ngram": 1 << 10,       # 11 address character n-gram
    "exact_name_heavy": 1 << 11, # 12 exact heavy normalized name
}

BLOCK_NAMES: List[str] = list(BLOCKS.keys())


@dataclass
class BlockingConfig:
    name_token_max_postings: int = 20000
    addr_token_max_postings: int = 20000
    ngram_max_postings: int = 8000
    rare_name_token_df: int = 400           # token is "rare" if df <= this
    rare_addr_token_df: int = 400
    min_token_len: int = 3
    ngram_n: int = 4
    ngram_max_grams: int = 40
    max_candidates_per_block: int = 200      # cap per S1 per retrieval block
    min_shared_tokens: int = 1
    postal_max_postings: int = 20000
    housenumber_max_postings: int = 20000


@dataclass
class BlockingIndex:
    """Holds every inverted index needed for candidate generation."""

    cs: CandidateSpace
    cfg: BlockingConfig
    stats: FrequencyStats
    country_to_gid: Dict[str, np.ndarray] = field(default_factory=dict)
    idx_exact_name: Dict[str, np.ndarray] = field(default_factory=dict)
    idx_core_name: Dict[str, np.ndarray] = field(default_factory=dict)
    idx_exact_name_heavy: Dict[str, np.ndarray] = field(default_factory=dict)
    idx_exact_addr: Dict[str, np.ndarray] = field(default_factory=dict)
    idx_name_houseno: Dict[str, np.ndarray] = field(default_factory=dict)
    idx_postal: Dict[str, np.ndarray] = field(default_factory=dict)
    idx_name_token: Dict[str, np.ndarray] = field(default_factory=dict)
    idx_addr_token: Dict[str, np.ndarray] = field(default_factory=dict)
    idx_name_ngram: Dict[str, np.ndarray] = field(default_factory=dict)
    idx_addr_ngram: Dict[str, np.ndarray] = field(default_factory=dict)
    idx_prefix: Dict[str, np.ndarray] = field(default_factory=dict)
    idf_name: Dict[str, float] = field(default_factory=dict)
    idf_addr: Dict[str, float] = field(default_factory=dict)

    # DuckDB index for large-scale streaming (optional)
    duckdb_index: Optional['idx.DuckDBIndex'] = field(default=None, repr=False)

    # ------------------------------------------------------------------ #
    def candidate_docs_for_tokens(
        self,
        tokens: Sequence[str],
        index: Dict[str, np.ndarray],
        max_postings: int,
        exclude: Optional[np.ndarray] = None,
    ) -> Dict[int, int]:
        """Return {global_id: shared_token_count} for tokens in ``index``."""
        counts: Dict[int, int] = {}
        for t in set(tokens):
            post = index.get(t)
            if post is None or len(post) > max_postings:
                continue
            for gid in post.tolist():
                counts[gid] = counts.get(gid, 0) + 1
        if exclude is not None and exclude.size:
            for gid in exclude.tolist():
                counts.pop(gid, None)
        return counts


def build_blocking_index(
    cs: CandidateSpace,
    stats: FrequencyStats,
    cfg: BlockingConfig,
    use_duckdb: bool = False,
    duckdb_path: Optional[str] = None,
) -> BlockingIndex:
    bi = BlockingIndex(cs=cs, cfg=cfg, stats=stats)

    # country partitions
    country_buckets: Dict[str, list] = {}
    for gid, c in enumerate(cs.country_norm):
        (country_buckets.setdefault(c, [])).append(gid)
    bi.country_to_gid = {c: np.asarray(v, dtype=np.int32) for c, v in country_buckets.items()}

    LOG.info("blocking: building exact-value indexes")
    bi.idx_exact_name = idx.build_value_index(cs.name_light)
    bi.idx_core_name = idx.build_value_index(cs.name_core)
    bi.idx_exact_name_heavy = idx.build_value_index(cs.name_heavy)
    bi.idx_exact_addr = idx.build_value_index(cs.addr_light)

    # name + house number composite key
    combo = [f"{cs.name_light[i]}|{cs.house_no[i]}" if cs.house_no[i] else "" for i in range(cs.total)]
    bi.idx_name_houseno = idx.build_value_index(combo)

    # postal index (explode multi-postal docs)
    pdocs: List[Tuple[str, ...]] = cs.postals
    bi.idx_postal = idx.build_inverted_index(
        pdocs, max_postings=cfg.postal_max_postings, min_doc_freq=1
    )

    # token indexes
    LOG.info("blocking: building name/address token indexes")
    bi.idx_name_token = idx.build_inverted_index(cs.name_tokens, max_postings=cfg.name_token_max_postings)
    bi.idx_addr_token = idx.build_inverted_index(cs.addr_tokens, max_postings=cfg.addr_token_max_postings)

    # character n-gram indexes
    LOG.info("blocking: building char n-gram indexes")
    bi.idx_name_ngram = idx.build_inverted_index(cs.name_char_ngrams, max_postings=cfg.ngram_max_postings)
    bi.idx_addr_ngram = idx.build_inverted_index(cs.addr_char_ngrams, max_postings=cfg.ngram_max_postings)

    LOG.info("blocking: building name prefix index")
    pref_docs = [
        (cs.name_light[i][:6],) if len(cs.name_light[i]) >= 4 else ()
        for i in range(cs.total)
    ]
    bi.idx_prefix = idx.build_inverted_index(pref_docs, max_postings=cfg.ngram_max_postings)

    # idf tables (only for tokens we will actually use)
    import math

    tn = stats.total_names or 1
    bi.idf_name = {t: math.log((1 + tn) / (1 + df)) + 1.0 for t, df in stats.name_token_freq.items()}
    ta = stats.total_addrs or 1
    bi.idf_addr = {t: math.log((1 + ta) / (1 + df)) + 1.0 for t, df in stats.addr_token_freq.items()}

    # Initialize DuckDB index if requested
    if use_duckdb and duckdb_path:
        bi.duckdb_index = idx.DuckDBIndex(duckdb_path)
        bi.duckdb_index.populate(cs)

    LOG.info(
        "blocking indexes: name_tok=%d addr_tok=%d name_ngram=%d addr_ngram=%d "
        "exact_name=%d exact_heavy=%d core_name=%d postal=%d",
        len(bi.idx_name_token),
        len(bi.idx_addr_token),
        len(bi.idx_name_ngram),
        len(bi.idx_addr_ngram),
        len(bi.idx_exact_name),
        len(bi.idx_exact_name_heavy),
        len(bi.idx_core_name),
        len(bi.idx_postal),
    )
    return bi