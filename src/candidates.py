"""Candidate generation: the multi-block union feeding the matcher.

For every Source-1 entity we retrieve candidates from up to 12 blocks and merge
them into a single candidate set, recording the provenance bitmask for each
candidate. Only this *final* candidate set is materialised (never an S1 x S2/S3
cross product).

Compatibility gates (country, name) are applied per block so that, for example,
a postal-only shared 5-digit number does not by itself import a candidate from a
different country with an unrelated name. This is a recall/precision trade-off
that we re-measure in ``candidates.evaluate_candidate_recall``.

NEW: Streaming/batch processing to avoid OOM on large datasets.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Iterator

import numpy as np
from rapidfuzz import fuzz

from . import indexing as idx
from .blocking import BLOCKS, BLOCK_NAMES, BlockingIndex, BlockingConfig
from .indexing import CandidateSpace, SourceTable
from .utils import LOG


@dataclass
class CandidatePair:
    s1_index: int                 # index in the S1 table
    gid: int                      # global candidate id in CandidateSpace
    blocks: int = 0               # provenance bitmask
    n_blocks: int = 0
    name_retrieval_rank: int = 10 ** 9
    addr_retrieval_rank: int = 10 ** 9
    ngram_retrieval_rank: int = 10 ** 9
    token_overlap: float = 0.0
    ngram_jaccard: float = 0.0
    idf_overlap: float = 0.0


def _capped(pairs: Iterable[Tuple[int, float]], cap: int) -> List[Tuple[int, float]]:
    pairs = list(pairs)
    if len(pairs) <= cap:
        return pairs
    pairs.sort(key=lambda x: -x[1])
    return pairs[:cap]


def generate_candidates(
    s1: SourceTable,
    bi: BlockingIndex,
    cfg: BlockingConfig,
    log_every: int = 20000,
    max_candidates_per_s1: int = 1000,
) -> List[CandidatePair]:
    """Generate the merged candidate set for every S1 entity."""
    cs = bi.cs
    out: List[CandidatePair] = []
    N1 = s1.n

    for i in range(N1):
        name_l = s1.name_light[i]
        name_c = s1.name_core[i]
        name_h = s1.name_heavy[i]
        name_toks = s1.name_tokens[i]
        core_toks = s1.name_core_tokens[i]
        name_ngrams = s1.name_char_ngrams[i]
        addr_l = s1.addr_light[i]
        addr_h = s1.addr_heavy[i]
        addr_toks = s1.addr_tokens[i]
        addr_ngrams = s1.addr_char_ngrams[i]
        postals = s1.postals[i]
        house = s1.house_no[i]
        country = s1.country_norm[i]

        acc: Dict[int, CandidatePair] = {}

        def add(gid: int, bit: int) -> CandidatePair:
            p = acc.get(gid)
            if p is None:
                p = CandidatePair(s1_index=i, gid=gid)
                acc[gid] = p
            p.blocks |= bit
            return p

        # ---- Block 1: country + normalized name (light) -------------------------
        if name_l:
            for gid in bi.idx_exact_name.get(name_l, ())[: cfg.max_candidates_per_block]:
                if country and cs.country_norm[gid] != country:
                    continue
                add(int(gid), BLOCKS["exact_name"])

        # ---- Block 1b: country + exact heavy normalized name --------------------
        if name_h:
            for gid in bi.idx_exact_name_heavy.get(name_h, ())[: cfg.max_candidates_per_block]:
                if country and cs.country_norm[gid] != country:
                    continue
                add(int(gid), BLOCKS["exact_name_heavy"])

        # ---- Block 2: country + core name ---------------------------------------
        if name_c:
            for gid in bi.idx_core_name.get(name_c, ())[: cfg.max_candidates_per_block]:
                if country and cs.country_norm[gid] != country:
                    continue
                add(int(gid), BLOCKS["core_name"])

        # ---- Block 3: exact normalized address ----------------------------------
        if addr_l:
            for gid in bi.idx_exact_addr.get(addr_l, ())[: cfg.max_candidates_per_block]:
                add(int(gid), BLOCKS["exact_address"])

        # ---- Block 4: name + house number ---------------------------------------
        if name_l and house:
            key = f"{name_l}|{house}"
            for gid in bi.idx_name_houseno.get(key, ())[: cfg.max_candidates_per_block]:
                add(int(gid), BLOCKS["name_housenumber"])

        # ---- Block 5: postal + compatible name ----------------------------------
        for pc in postals:
            post = bi.idx_postal.get(pc)
            if post is None or len(post) > cfg.postal_max_postings:
                continue
            kept = []
            for gid in post.tolist():
                cname = cs.name_light[gid]
                if name_l and cname:
                    sim = fuzz.token_set_ratio(name_l, cname) / 100.0
                    if sim < 0.5:
                        continue
                    kept.append((gid, sim))
                else:
                    kept.append((gid, 0.4))
            for gid, _ in _capped(kept, cfg.max_candidates_per_block):
                add(int(gid), BLOCKS["postal_name"])

        # ---- Block 6: rare business-name token ----------------------------------
        rare_name_gids: Dict[int, float] = {}
        for t in set(core_toks):
            if len(t) < cfg.min_token_len:
                continue
            df = bi.stats.name_token_freq.get(t, 0)
            if df == 0 or df > cfg.rare_name_token_df:
                continue
            post = bi.idx_name_token.get(t)
            if post is None:
                continue
            w = bi.idf_name.get(t, 1.0)
            for gid in post.tolist():
                rare_name_gids[gid] = rare_name_gids.get(gid, 0.0) + w
        for gid, _ in _capped(list(rare_name_gids.items()), cfg.max_candidates_per_block):
            add(int(gid), BLOCKS["rare_name_token"])

        # ---- Block 7: rare address token ----------------------------------------
        rare_addr_gids: Dict[int, float] = {}
        for t in set(addr_toks):
            if len(t) < cfg.min_token_len or t.isdigit():
                continue
            df = bi.stats.addr_token_freq.get(t, 0)
            if df == 0 or df > cfg.rare_addr_token_df:
                continue
            post = bi.idx_addr_token.get(t)
            if post is None:
                continue
            w = bi.idf_addr.get(t, 1.0)
            for gid in post.tolist():
                rare_addr_gids[gid] = rare_addr_gids.get(gid, 0.0) + w
        for gid, _ in _capped(list(rare_addr_gids.items()), cfg.max_candidates_per_block):
            add(int(gid), BLOCKS["rare_addr_token"])

        # ---- Block 8: name character n-gram retrieval ---------------------------
        if name_ngrams:
            ng_counts: Dict[int, int] = {}
            for g in name_ngrams:
                post = bi.idx_name_ngram.get(g)
                if post is None:
                    continue
                for gid in post.tolist():
                    ng_counts[gid] = ng_counts.get(gid, 0) + 1
            cand = [(g, c) for g, c in ng_counts.items() if c >= 2]
            cand.sort(key=lambda x: -x[1])
            for rank, (gid, cnt) in enumerate(cand[: cfg.max_candidates_per_block]):
                p = add(int(gid), BLOCKS["char_ngram"])
                p.ngram_retrieval_rank = min(p.ngram_retrieval_rank, rank)
                p.ngram_jaccard = cnt / max(1, len(name_ngrams))

        # ---- Block 8b: address character n-gram retrieval -----------------------
        if addr_ngrams:
            ng_counts: Dict[int, int] = {}
            for g in addr_ngrams:
                post = bi.idx_addr_ngram.get(g)
                if post is None:
                    continue
                for gid in post.tolist():
                    ng_counts[gid] = ng_counts.get(gid, 0) + 1
            cand = [(g, c) for g, c in ng_counts.items() if c >= 2]
            cand.sort(key=lambda x: -x[1])
            for rank, (gid, cnt) in enumerate(cand[: cfg.max_candidates_per_block]):
                p = add(int(gid), BLOCKS["addr_ngram"])
                p.addr_retrieval_rank = min(p.addr_retrieval_rank, rank)

        # ---- Block 9: token-overlap retrieval -----------------------------------
        tok_counts: Dict[int, int] = {}
        for t in set(core_toks):
            if len(t) < cfg.min_token_len:
                continue
            post = bi.idx_name_token.get(t)
            if post is None or len(post) > cfg.name_token_max_postings:
                continue
            for gid in post.tolist():
                tok_counts[gid] = tok_counts.get(gid, 0) + 1
        cand9 = [
            (g, c / max(1, len(set(core_toks))))
            for g, c in tok_counts.items()
            if c >= cfg.min_shared_tokens
        ]
        cand9.sort(key=lambda x: -x[1])
        for rank, (gid, ov) in enumerate(cand9[: cfg.max_candidates_per_block]):
            p = add(int(gid), BLOCKS["token_overlap"])
            p.token_overlap = max(p.token_overlap, ov)
            p.name_retrieval_rank = min(p.name_retrieval_rank, rank)

        # ---- Block 10: country-aware approximate retrieval ----------------------
        if name_l:
            partition = bi.country_to_gid.get(country)
            if partition is not None and partition.size and partition.size <= 200000:
                scored = []
                # only consider docs that already share a token or ngram (cheap gate)
                pool = set(tok_counts.keys())
                if name_ngrams:
                    ng_counts = {}
                    for g in name_ngrams:
                        post = bi.idx_name_ngram.get(g)
                        if post is not None:
                            for gid in post.tolist():
                                ng_counts[gid] = ng_counts.get(gid, 0) + 1
                    pool |= set(ng_counts.keys())
                for gid in pool:
                    if cs.country_norm[gid] != country:
                        continue
                    sim = fuzz.token_set_ratio(name_l, cs.name_light[gid]) / 100.0
                    if sim >= 0.60:
                        scored.append((gid, sim))
                scored.sort(key=lambda x: -x[1])
                for gid, _ in scored[: cfg.max_candidates_per_block]:
                    add(int(gid), BLOCKS["country_approx"])

        # finalise metrics
        for p in acc.values():
            p.n_blocks = int(bin(p.blocks).count("1"))
            ctoks = set(cs.name_core_tokens[p.gid])
            if core_toks and ctoks:
                shared = set(core_toks) & ctoks
                p.idf_overlap = sum(bi.idf_name.get(t, 1.0) for t in shared)
        out.extend(acc.values())

        if log_every and (i + 1) % log_every == 0:
            LOG.info("  candidates: %d/%d S1 processed (%d pairs)", i + 1, N1, len(out))

    return out


# --------------------------------------------------------------------------- #
# Streaming candidate generation (batched for memory efficiency)
# --------------------------------------------------------------------------- #
def generate_candidates_streaming(
    s1: SourceTable,
    bi: BlockingIndex,
    cfg: BlockingConfig,
    batch_size: int = 25000,
    max_candidates_per_s1: int = 1000,
    log_every: int = 5000,
) -> Iterator[List[CandidatePair]]:
    """Generate candidates in batches to avoid OOM on large datasets.
    
    Yields lists of CandidatePair for each batch of S1 entities.
    """
    cs = bi.cs
    N1 = s1.n
    for batch_start in range(0, N1, batch_size):
        batch_end = min(batch_start + batch_size, N1)
        batch_pairs: List[CandidatePair] = []
        
        for i in range(batch_start, batch_end):
            name_l = s1.name_light[i]
            name_c = s1.name_core[i]
            name_h = s1.name_heavy[i]
            name_toks = s1.name_tokens[i]
            core_toks = s1.name_core_tokens[i]
            name_ngrams = s1.name_char_ngrams[i]
            addr_l = s1.addr_light[i]
            addr_h = s1.addr_heavy[i]
            addr_toks = s1.addr_tokens[i]
            addr_ngrams = s1.addr_char_ngrams[i]
            postals = s1.postals[i]
            house = s1.house_no[i]
            country = s1.country_norm[i]

            acc: Dict[int, CandidatePair] = {}

            def add(gid: int, bit: int) -> CandidatePair:
                p = acc.get(gid)
                if p is None:
                    p = CandidatePair(s1_index=i, gid=gid)
                    acc[gid] = p
                p.blocks |= bit
                return p

            # Block 1: country + normalized name (light)
            if name_l:
                for gid in bi.idx_exact_name.get(name_l, ())[: cfg.max_candidates_per_block]:
                    if country and cs.country_norm[gid] != country:
                        continue
                    add(int(gid), BLOCKS["exact_name"])

            # Block 1b: country + exact heavy normalized name
            if name_h:
                for gid in bi.idx_exact_name_heavy.get(name_h, ())[: cfg.max_candidates_per_block]:
                    if country and cs.country_norm[gid] != country:
                        continue
                    add(int(gid), BLOCKS["exact_name_heavy"])

            # Block 2: country + core name
            if name_c:
                for gid in bi.idx_core_name.get(name_c, ())[: cfg.max_candidates_per_block]:
                    if country and cs.country_norm[gid] != country:
                        continue
                    add(int(gid), BLOCKS["core_name"])

            # Block 3: exact normalized address
            if addr_l:
                for gid in bi.idx_exact_addr.get(addr_l, ())[: cfg.max_candidates_per_block]:
                    add(int(gid), BLOCKS["exact_address"])

            # Block 4: name + house number
            if name_l and house:
                key = f"{name_l}|{house}"
                for gid in bi.idx_name_houseno.get(key, ())[: cfg.max_candidates_per_block]:
                    add(int(gid), BLOCKS["name_housenumber"])

            # Block 5: postal + compatible name
            for pc in postals:
                post = bi.idx_postal.get(pc)
                if post is None or len(post) > cfg.postal_max_postings:
                    continue
                kept = []
                for gid in post.tolist():
                    cname = cs.name_light[gid]
                    if name_l and cname:
                        sim = fuzz.token_set_ratio(name_l, cname) / 100.0
                        if sim < 0.5:
                            continue
                        kept.append((gid, sim))
                    else:
                        kept.append((gid, 0.4))
                for gid, _ in _capped(kept, cfg.max_candidates_per_block):
                    add(int(gid), BLOCKS["postal_name"])

            # Block 6: rare business-name token
            rare_name_gids: Dict[int, float] = {}
            for t in set(core_toks):
                if len(t) < cfg.min_token_len:
                    continue
                df = bi.stats.name_token_freq.get(t, 0)
                if df == 0 or df > cfg.rare_name_token_df:
                    continue
                post = bi.idx_name_token.get(t)
                if post is None:
                    continue
                w = bi.idf_name.get(t, 1.0)
                for gid in post.tolist():
                    rare_name_gids[gid] = rare_name_gids.get(gid, 0.0) + w
            for gid, _ in _capped(list(rare_name_gids.items()), cfg.max_candidates_per_block):
                add(int(gid), BLOCKS["rare_name_token"])

            # Block 7: rare address token
            rare_addr_gids: Dict[int, float] = {}
            for t in set(addr_toks):
                if len(t) < cfg.min_token_len or t.isdigit():
                    continue
                df = bi.stats.addr_token_freq.get(t, 0)
                if df == 0 or df > cfg.rare_addr_token_df:
                    continue
                post = bi.idx_addr_token.get(t)
                if post is None:
                    continue
                w = bi.idf_addr.get(t, 1.0)
                for gid in post.tolist():
                    rare_addr_gids[gid] = rare_addr_gids.get(gid, 0.0) + w
            for gid, _ in _capped(list(rare_addr_gids.items()), cfg.max_candidates_per_block):
                add(int(gid), BLOCKS["rare_addr_token"])

            # Block 8: name character n-gram retrieval
            if name_ngrams:
                ng_counts: Dict[int, int] = {}
                for g in name_ngrams:
                    post = bi.idx_name_ngram.get(g)
                    if post is None:
                        continue
                    for gid in post.tolist():
                        ng_counts[gid] = ng_counts.get(gid, 0) + 1
                cand = [(g, c) for g, c in ng_counts.items() if c >= 2]
                cand.sort(key=lambda x: -x[1])
                for rank, (gid, cnt) in enumerate(cand[: cfg.max_candidates_per_block]):
                    p = add(int(gid), BLOCKS["char_ngram"])
                    p.ngram_retrieval_rank = min(p.ngram_retrieval_rank, rank)
                    p.ngram_jaccard = cnt / max(1, len(name_ngrams))

            # Block 8b: address character n-gram retrieval
            if addr_ngrams:
                ng_counts: Dict[int, int] = {}
                for g in addr_ngrams:
                    post = bi.idx_addr_ngram.get(g)
                    if post is None:
                        continue
                    for gid in post.tolist():
                        ng_counts[gid] = ng_counts.get(gid, 0) + 1
                cand = [(g, c) for g, c in ng_counts.items() if c >= 2]
                cand.sort(key=lambda x: -x[1])
                for rank, (gid, cnt) in enumerate(cand[: cfg.max_candidates_per_block]):
                    p = add(int(gid), BLOCKS["addr_ngram"])
                    p.addr_retrieval_rank = min(p.addr_retrieval_rank, rank)

            # Block 9: token-overlap retrieval
            tok_counts: Dict[int, int] = {}
            for t in set(core_toks):
                if len(t) < cfg.min_token_len:
                    continue
                post = bi.idx_name_token.get(t)
                if post is None or len(post) > cfg.name_token_max_postings:
                    continue
                for gid in post.tolist():
                    tok_counts[gid] = tok_counts.get(gid, 0) + 1
            cand9 = [
                (g, c / max(1, len(set(core_toks))))
                for g, c in tok_counts.items()
                if c >= cfg.min_shared_tokens
            ]
            cand9.sort(key=lambda x: -x[1])
            for rank, (gid, ov) in enumerate(cand9[: cfg.max_candidates_per_block]):
                p = add(int(gid), BLOCKS["token_overlap"])
                p.token_overlap = max(p.token_overlap, ov)
                p.name_retrieval_rank = min(p.name_retrieval_rank, rank)

            # Block 10: country-aware approximate retrieval
            if name_l:
                partition = bi.country_to_gid.get(country)
                if partition is not None and partition.size and partition.size <= 200000:
                    scored = []
                    pool = set(tok_counts.keys())
                    if name_ngrams:
                        ng_counts = {}
                        for g in name_ngrams:
                            post = bi.idx_name_ngram.get(g)
                            if post is not None:
                                for gid in post.tolist():
                                    ng_counts[gid] = ng_counts.get(gid, 0) + 1
                        pool |= set(ng_counts.keys())
                    for gid in pool:
                        if cs.country_norm[gid] != country:
                            continue
                        sim = fuzz.token_set_ratio(name_l, cs.name_light[gid]) / 100.0
                        if sim >= 0.60:
                            scored.append((gid, sim))
                    scored.sort(key=lambda x: -x[1])
                    for gid, _ in scored[: cfg.max_candidates_per_block]:
                        add(int(gid), BLOCKS["country_approx"])

            # finalise metrics for this entity
            for p in acc.values():
                p.n_blocks = int(bin(p.blocks).count("1"))
                ctoks = set(cs.name_core_tokens[p.gid])
                if core_toks and ctoks:
                    shared = set(core_toks) & ctoks
                    p.idf_overlap = sum(bi.idf_name.get(t, 1.0) for t in shared)

            # Cap candidates per S1 if too many
            entity_pairs = list(acc.values())
            if len(entity_pairs) > max_candidates_per_s1:
                # Sort by n_blocks descending, then by idf_overlap
                entity_pairs.sort(key=lambda x: (-x.n_blocks, -x.idf_overlap))
                entity_pairs = entity_pairs[:max_candidates_per_s1]
            
            batch_pairs.extend(entity_pairs)

        if log_every and (batch_end % log_every == 0 or batch_end == N1):
            LOG.info("  candidates: %d/%d S1 processed (%d pairs in batch)", batch_end, N1, len(batch_pairs))

        yield batch_pairs


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
@dataclass
class RecallReport:
    n_s1: int
    n_pos_pairs: int
    n_pos_pairs_found: int
    candidate_recall: float
    n_candidates: int
    reduction_ratio: float
    mean_candidates: float
    median_candidates: float
    p95_candidates: float
    p99_candidates: float
    max_candidates: float
    by_source: Dict[str, Dict[str, float]] = field(default_factory=dict)
    by_country: Dict[str, Dict[str, float]] = field(default_factory=dict)
    per_entity_counts: List[int] = field(default_factory=list)

    def as_dict(self) -> dict:
        d = dict(self.__dict__)
        # do not serialise the long per-entity list into JSON reports
        d.pop("per_entity_counts", None)
        return d


def evaluate_candidate_recall(
    s1: SourceTable,
    cs: CandidateSpace,
    pairs: Sequence[CandidatePair],
    positives: Dict[int, set],          # s1_index -> set(gid)
    n1_total: Optional[int] = None,
) -> Tuple[RecallReport, np.ndarray]:
    """Compute candidate recall / reduction and per-entity candidate counts."""
    counts = np.zeros(s1.n, dtype=np.int64)
    found: Dict[int, set] = {}
    for p in pairs:
        counts[p.s1_index] += 1
        if p.s1_index in positives and p.gid in positives[p.s1_index]:
            found.setdefault(p.s1_index, set()).add(p.gid)

    n_pos = sum(len(v) for v in positives.values())
    n_found = sum(len(v) for v in found.values())
    n_cand = len(pairs)
    n_possible = s1.n * cs.total if cs.total else 1

    per_entity = counts.tolist()
    nonzero = counts[counts > 0]
    rep = RecallReport(
        n_s1=s1.n,
        n_pos_pairs=n_pos,
        n_pos_pairs_found=n_found,
        candidate_recall=(n_found / n_pos) if n_pos else 1.0,
        n_candidates=n_cand,
        reduction_ratio=(1.0 - n_cand / max(1, n_possible)),
        mean_candidates=float(counts.mean()) if s1.n else 0.0,
        median_candidates=float(np.median(counts)) if s1.n else 0.0,
        p95_candidates=float(np.percentile(counts, 95)) if s1.n else 0.0,
        p99_candidates=float(np.percentile(counts, 99)) if s1.n else 0.0,
        max_candidates=float(counts.max()) if s1.n else 0.0,
        per_entity_counts=per_entity,
    )

    # recall split by source of the true match
    src_names = cs.order
    src_hit = {s: [0, 0] for s in src_names}
    for si, gids in positives.items():
        for g in gids:
            s = src_names[int(cs.source_of[g])]
            src_hit[s][1] += 1
            if g in found.get(si, ()):
                src_hit[s][0] += 1
    rep.by_source = {
        s: {
            "found": v[0],
            "total": v[1],
            "recall": (v[0] / v[1]) if v[1] else 1.0,
        }
        for s, v in src_hit.items()
    }

    # recall split by country
    country_hit: Dict[str, List[int]] = {}
    for si, gids in positives.items():
        c = s1.country_norm[si] or "(empty)"
        ch = country_hit.setdefault(c, [0, 0])
        for g in gids:
            ch[1] += 1
            if g in found.get(si, ()):
                ch[0] += 1
    rep.by_country = {
        c: {"found": v[0], "total": v[1], "recall": (v[0] / v[1]) if v[1] else 1.0}
        for c, v in sorted(country_hit.items(), key=lambda kv: -kv[1][1])
    }
    return rep, counts


# --------------------------------------------------------------------------- #
# Candidate recall diagnostic per block
# --------------------------------------------------------------------------- #
@dataclass
class BlockRecallReport:
    block_name: str
    positive_pairs_found: int
    positive_recall: float
    candidate_count: int
    mean_candidates_per_s1: float
    p95_candidates: float
    p99_candidates: float


def block_recall_diagnostic(
    s1: SourceTable,
    cs: CandidateSpace,
    bi: BlockingIndex,
    cfg: BlockingConfig,
    positives: Dict[int, set],
    sample_size: Optional[int] = None,
) -> List[BlockRecallReport]:
    """Run candidate recall diagnostic for each block independently.
    
    This helps identify which blocks contribute most to recall and their cost.
    """
    from collections import defaultdict
    
    N1 = s1.n if sample_size is None else min(s1.n, sample_size)
    reports = []
    
    # Collect all candidate pairs per block
    block_pairs: Dict[str, List[CandidatePair]] = defaultdict(list)
    block_counts: Dict[str, np.ndarray] = defaultdict(lambda: np.zeros(N1, dtype=np.int64))
    block_found: Dict[str, Dict[int, set]] = defaultdict(lambda: defaultdict(set))
    
    for i in range(N1):
        name_l = s1.name_light[i]
        name_c = s1.name_core[i]
        name_h = s1.name_heavy[i]
        name_toks = s1.name_tokens[i]
        core_toks = s1.name_core_tokens[i]
        name_ngrams = s1.name_char_ngrams[i]
        addr_l = s1.addr_light[i]
        addr_toks = s1.addr_tokens[i]
        addr_ngrams = s1.addr_char_ngrams[i]
        postals = s1.postals[i]
        house = s1.house_no[i]
        country = s1.country_norm[i]
        
        # Test each block independently
        blocks_to_test = [
            ("exact_name", lambda: _block_exact_name(i, name_l, country, bi, cs, cfg)),
            ("core_name", lambda: _block_core_name(i, name_c, country, bi, cs, cfg)),
            ("exact_name_heavy", lambda: _block_exact_name_heavy(i, name_h, country, bi, cs, cfg)),
            ("exact_address", lambda: _block_exact_addr(i, addr_l, bi, cs, cfg)),
            ("name_housenumber", lambda: _block_name_houseno(i, name_l, house, bi, cs, cfg)),
            ("postal_name", lambda: _block_postal_name(i, name_l, postals, bi, cs, cfg)),
            ("rare_name_token", lambda: _block_rare_name(i, core_toks, bi, cs, cfg)),
            ("rare_addr_token", lambda: _block_rare_addr(i, addr_toks, bi, cs, cfg)),
            ("char_ngram", lambda: _block_char_ngram(i, name_ngrams, bi, cs, cfg)),
            ("addr_ngram", lambda: _block_addr_ngram(i, addr_ngrams, bi, cs, cfg)),
            ("token_overlap", lambda: _block_token_overlap(i, core_toks, bi, cs, cfg)),
            ("country_approx", lambda: _block_country_approx(i, name_l, country, bi, cs, cfg, core_toks)),
        ]
        
        for block_name, block_fn in blocks_to_test:
            cands = block_fn()
            for gid in cands:
                block_pairs[block_name].append(CandidatePair(s1_index=i, gid=gid, blocks=BLOCKS[block_name]))
                block_counts[block_name][i] += 1
                if i in positives and gid in positives[i]:
                    block_found[block_name][i].add(gid)
    
    n_pos = sum(len(v) for v in positives.values() if v and min(v) < N1)
    
    for block_name in BLOCK_NAMES:
        pairs = block_pairs.get(block_name, [])
        counts = block_counts.get(block_name, np.zeros(N1, dtype=np.int64))
        found = block_found.get(block_name, {})
        
        n_found = sum(len(v) for v in found.values())
        n_cand = len(pairs)
        
        if n_cand > 0:
            nonzero = counts[counts > 0]
            mean_cand = float(counts.mean()) if N1 else 0.0
            p95 = float(np.percentile(counts, 95)) if N1 else 0.0
            p99 = float(np.percentile(counts, 99)) if N1 else 0.0
        else:
            mean_cand = p95 = p99 = 0.0
        
        recall = n_found / n_pos if n_pos > 0 else 1.0
        
        reports.append(BlockRecallReport(
            block_name=block_name,
            positive_pairs_found=n_found,
            positive_recall=recall,
            candidate_count=n_cand,
            mean_candidates_per_s1=mean_cand,
            p95_candidates=p95,
            p99_candidates=p99,
        ))
    
    return reports


# Helper functions for block recall diagnostic
def _block_exact_name(i, name_l, country, bi, cs, cfg):
    if not name_l:
        return []
    cands = []
    for gid in bi.idx_exact_name.get(name_l, ())[: cfg.max_candidates_per_block]:
        if country and cs.country_norm[gid] != country:
            continue
        cands.append(int(gid))
    return cands

def _block_core_name(i, name_c, country, bi, cs, cfg):
    if not name_c:
        return []
    cands = []
    for gid in bi.idx_core_name.get(name_c, ())[: cfg.max_candidates_per_block]:
        if country and cs.country_norm[gid] != country:
            continue
        cands.append(int(gid))
    return cands

def _block_exact_name_heavy(i, name_h, country, bi, cs, cfg):
    if not name_h:
        return []
    cands = []
    for gid in bi.idx_exact_name_heavy.get(name_h, ())[: cfg.max_candidates_per_block]:
        if country and cs.country_norm[gid] != country:
            continue
        cands.append(int(gid))
    return cands

def _block_exact_addr(i, addr_l, bi, cs, cfg):
    if not addr_l:
        return []
    return [int(gid) for gid in bi.idx_exact_addr.get(addr_l, ())[: cfg.max_candidates_per_block]]

def _block_name_houseno(i, name_l, house, bi, cs, cfg):
    if not name_l or not house:
        return []
    key = f"{name_l}|{house}"
    return [int(gid) for gid in bi.idx_name_houseno.get(key, ())[: cfg.max_candidates_per_block]]

def _block_postal_name(i, name_l, postals, bi, cs, cfg):
    cands = []
    for pc in postals:
        post = bi.idx_postal.get(pc)
        if post is None or len(post) > cfg.postal_max_postings:
            continue
        kept = []
        for gid in post.tolist():
            cname = cs.name_light[gid]
            if name_l and cname:
                sim = fuzz.token_set_ratio(name_l, cname) / 100.0
                if sim < 0.5:
                    continue
                kept.append((gid, sim))
            else:
                kept.append((gid, 0.4))
        for gid, _ in _capped(kept, cfg.max_candidates_per_block):
            cands.append(gid)
    return cands

def _block_rare_name(i, core_toks, bi, cs, cfg):
    rare_name_gids: Dict[int, float] = {}
    for t in set(core_toks):
        if len(t) < cfg.min_token_len:
            continue
        df = bi.stats.name_token_freq.get(t, 0)
        if df == 0 or df > cfg.rare_name_token_df:
            continue
        post = bi.idx_name_token.get(t)
        if post is None:
            continue
        w = bi.idf_name.get(t, 1.0)
        for gid in post.tolist():
            rare_name_gids[gid] = rare_name_gids.get(gid, 0.0) + w
    return [gid for gid, _ in _capped(list(rare_name_gids.items()), cfg.max_candidates_per_block)]

def _block_rare_addr(i, addr_toks, bi, cs, cfg):
    rare_addr_gids: Dict[int, float] = {}
    for t in set(addr_toks):
        if len(t) < cfg.min_token_len or t.isdigit():
            continue
        df = bi.stats.addr_token_freq.get(t, 0)
        if df == 0 or df > cfg.rare_addr_token_df:
            continue
        post = bi.idx_addr_token.get(t)
        if post is None:
            continue
        w = bi.idf_addr.get(t, 1.0)
        for gid in post.tolist():
            rare_addr_gids[gid] = rare_addr_gids.get(gid, 0.0) + w
    return [gid for gid, _ in _capped(list(rare_addr_gids.items()), cfg.max_candidates_per_block)]

def _block_char_ngram(i, name_ngrams, bi, cs, cfg):
    if not name_ngrams:
        return []
    ng_counts: Dict[int, int] = {}
    for g in name_ngrams:
        post = bi.idx_name_ngram.get(g)
        if post is None:
            continue
        for gid in post.tolist():
            ng_counts[gid] = ng_counts.get(gid, 0) + 1
    cand = [(g, c) for g, c in ng_counts.items() if c >= 2]
    cand.sort(key=lambda x: -x[1])
    return [gid for gid, _ in cand[: cfg.max_candidates_per_block]]

def _block_addr_ngram(i, addr_ngrams, bi, cs, cfg):
    if not addr_ngrams:
        return []
    ng_counts: Dict[int, int] = {}
    for g in addr_ngrams:
        post = bi.idx_addr_ngram.get(g)
        if post is None:
            continue
        for gid in post.tolist():
            ng_counts[gid] = ng_counts.get(gid, 0) + 1
    cand = [(g, c) for g, c in ng_counts.items() if c >= 2]
    cand.sort(key=lambda x: -x[1])
    return [gid for gid, _ in cand[: cfg.max_candidates_per_block]]

def _block_token_overlap(i, core_toks, bi, cs, cfg):
    tok_counts: Dict[int, int] = {}
    for t in set(core_toks):
        if len(t) < cfg.min_token_len:
            continue
        post = bi.idx_name_token.get(t)
        if post is None or len(post) > cfg.name_token_max_postings:
            continue
        for gid in post.tolist():
            tok_counts[gid] = tok_counts.get(gid, 0) + 1
    cand9 = [
        (g, c / max(1, len(set(core_toks))))
        for g, c in tok_counts.items()
        if c >= cfg.min_shared_tokens
    ]
    cand9.sort(key=lambda x: -x[1])
    return [gid for gid, _ in cand9[: cfg.max_candidates_per_block]]

def _block_country_approx(i, name_l, country, bi, cs, cfg, core_toks=None):
    if not name_l:
        return []
    partition = bi.country_to_gid.get(country)
    if partition is None or not partition.size or partition.size > 200000:
        return []
    scored = []
    tok_counts = {}
    if core_toks is not None:
        for t in set(core_toks):
            if len(t) < cfg.min_token_len:
                continue
            post = bi.idx_name_token.get(t)
            if post is None or len(post) > cfg.name_token_max_postings:
                continue
            for gid in post.tolist():
                tok_counts[gid] = tok_counts.get(gid, 0) + 1
    pool = set(tok_counts.keys())
    for gid in pool:
        if cs.country_norm[gid] != country:
            continue
        sim = fuzz.token_set_ratio(name_l, cs.name_light[gid]) / 100.0
        if sim >= 0.60:
            scored.append((gid, sim))
    scored.sort(key=lambda x: -x[1])
    return [gid for gid, _ in scored[: cfg.max_candidates_per_block]]