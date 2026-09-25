"""Indexing: compact per-source tables plus inverted indexes for blocking.

Everything here is designed for constrained RAM:

* entity ids / normalized strings are Python str held once (unavoidable, but we
  never duplicate them per-pair);
* tokens are stored as ``list[tuple[str, ...]]`` (tuples are lighter than re-split
  strings and hash faster);
* posting lists are ``int32`` numpy arrays, not Python lists of ints;
* frequency statistics are computed with ``collections.Counter`` over the same
  token structures (single pass, no dense matrices).
"""
from __future__ import annotations

import array
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import polars as pl
from rapidfuzz import fuzz

from . import normalization as norm
from .utils import LOG

TokenSeq = Tuple[str, ...]


# --------------------------------------------------------------------------- #
# Per-source table
# --------------------------------------------------------------------------- #
@dataclass
class SourceTable:
    """All representations for a single source (S1/S2/S3)."""

    tag: str
    entity_ids: List[str]
    n: int = 0

    name_raw: List[str] = field(default_factory=list)
    name_light: List[str] = field(default_factory=list)
    name_heavy: List[str] = field(default_factory=list)
    name_core: List[str] = field(default_factory=list)
    name_tokens: List[TokenSeq] = field(default_factory=list)
    name_core_tokens: List[TokenSeq] = field(default_factory=list)

    addr_raw: List[str] = field(default_factory=list)
    addr_light: List[str] = field(default_factory=list)
    addr_heavy: List[str] = field(default_factory=list)
    addr_tokens: List[TokenSeq] = field(default_factory=list)
    addr_numbers: List[TokenSeq] = field(default_factory=list)
    postals: List[TokenSeq] = field(default_factory=list)
    house_no: List[str] = field(default_factory=list)

    country_raw: List[str] = field(default_factory=list)
    country_norm: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.n = len(self.entity_ids)

    def id_to_idx(self) -> Dict[str, int]:
        return {e: i for i, e in enumerate(self.entity_ids)}

    def subset_indices(self, indices: Sequence[int]) -> List[str]:
        return [self.entity_ids[i] for i in indices]


def build_source_table(tag: str, df: "pl.DataFrame") -> SourceTable:
    """Convert a standardized polars frame into a SourceTable of representations."""
    entity_ids = df["entity_id"].fill_null("").cast(pl.Utf8).to_list()
    names = df["business_name"].cast(pl.Utf8).to_list()
    addrs = df["business_address"].cast(pl.Utf8).to_list()
    countries = df["country"].cast(pl.Utf8).to_list()

    st = SourceTable(tag=tag, entity_ids=entity_ids)
    for i in range(len(entity_ids)):
        nm = names[i]
        ad = addrs[i]
        ct = countries[i]
        nr = norm.TextRepresenter.name_reprs(nm)
        ar = norm.TextRepresenter.address_reprs(ad)
        cr = norm.TextRepresenter.country_repr(ct)
        st.name_raw.append(nm or "")
        st.name_light.append(nr["business_name_light_normalized"])
        st.name_heavy.append(nr["business_name_heavy_normalized"])
        st.name_core.append(nr["business_name_core"])
        st.name_tokens.append(tuple(nr["business_name_tokens"]))
        st.name_core_tokens.append(tuple(nr["business_name_core_tokens"]))
        st.addr_raw.append(ad or "")
        st.addr_light.append(ar["business_address_normalized"])
        st.addr_heavy.append(ar["business_address_heavy"])
        st.addr_tokens.append(tuple(ar["business_address_tokens"]))
        st.addr_numbers.append(tuple(ar["address_numbers"]))
        st.postals.append(tuple(ar["postal_candidates"]))
        st.house_no.append(ar["house_number"])
        st.country_raw.append(ct or "")
        st.country_norm.append(cr["country_normalized"])
    return st


# --------------------------------------------------------------------------- #
# Inverted index
# --------------------------------------------------------------------------- #
def build_inverted_index(
    docs: Sequence[TokenSeq],
    max_postings: Optional[int] = None,
    min_doc_freq: int = 1,
) -> Dict[str, np.ndarray]:
    """token -> sorted int32 array of doc indices.

    ``max_postings`` drops ultra-frequent tokens (a standard blocking cost cap).
    """
    buckets: Dict[str, array.array] = {}
    for i, toks in enumerate(docs):
        for t in set(toks):
            b = buckets.get(t)
            if b is None:
                b = array.array("i")
                buckets[t] = b
            b.append(i)

    out: Dict[str, np.ndarray] = {}
    for t, b in buckets.items():
        if len(b) < min_doc_freq:
            continue
        if max_postings is not None and len(b) > max_postings:
            continue
        out[t] = np.frombuffer(b, dtype=np.int32)
    return out


def build_value_index(values: Sequence[str], min_count: int = 1) -> Dict[str, np.ndarray]:
    """Exact-value index (e.g. normalized names, addresses) -> doc indices."""
    buckets: Dict[str, array.array] = {}
    for i, v in enumerate(values):
        if not v:
            continue
        b = buckets.get(v)
        if b is None:
            b = array.array("i")
            buckets[v] = b
        b.append(i)
    return {
        v: np.frombuffer(b, dtype=np.int32)
        for v, b in buckets.items()
        if len(b) >= min_count
    }


def char_ngrams(text: str, n: int = 4, max_grams: int = 40) -> TokenSeq:
    """Character n-grams over the padded compact string (deduplicated)."""
    s = text.replace(" ", "")
    if len(s) < n:
        return tuple([s]) if s else ()
    grams = [s[i : i + n] for i in range(len(s) - n + 1)]
    if len(grams) > max_grams:
        # keep a spread of grams (head+tail) deterministically
        step = len(grams) / max_grams
        grams = [grams[int(i * step)] for i in range(max_grams)]
    return tuple(dict.fromkeys(grams))


def build_ngram_docs(name_light: Sequence[str], n: int = 4, max_grams: int = 40) -> List[TokenSeq]:
    return [char_ngrams(s, n=n, max_grams=max_grams) for s in name_light]


# --------------------------------------------------------------------------- #
# Frequency / rarity statistics
# --------------------------------------------------------------------------- #
@dataclass
class FrequencyStats:
    name_freq: Dict[str, int]
    core_name_freq: Dict[str, int]
    addr_freq: Dict[str, int]
    name_token_freq: Dict[str, int]
    addr_token_freq: Dict[str, int]
    total_names: int
    total_addrs: int

    def idf_name_token(self, tok: str) -> float:
        import math

        df = self.name_token_freq.get(tok, 0)
        return math.log((1 + self.total_names) / (1 + df)) + 1.0

    def idf_addr_token(self, tok: str) -> float:
        import math

        df = self.addr_token_freq.get(tok, 0)
        return math.log((1 + self.total_addrs) / (1 + df)) + 1.0


def build_frequency_stats(tables: Iterable[SourceTable]) -> FrequencyStats:
    """Frequency stats over the *training* corpus only (no test leakage)."""
    name_freq: Dict[str, int] = {}
    core_freq: Dict[str, int] = {}
    addr_freq: Dict[str, int] = {}
    name_tok: Dict[str, int] = {}
    addr_tok: Dict[str, int] = {}
    total_names = 0
    total_addrs = 0

    for st in tables:
        for i in range(st.n):
            nm = st.name_light[i]
            if nm:
                name_freq[nm] = name_freq.get(nm, 0) + 1
                total_names += 1
            cn = st.name_core[i]
            if cn:
                core_freq[cn] = core_freq.get(cn, 0) + 1
            ad = st.addr_light[i]
            if ad:
                addr_freq[ad] = addr_freq.get(ad, 0) + 1
                total_addrs += 1
            for t in set(st.name_tokens[i]):
                name_tok[t] = name_tok.get(t, 0) + 1
            for t in set(st.addr_tokens[i]):
                addr_tok[t] = addr_tok.get(t, 0) + 1

    return FrequencyStats(
        name_freq=name_freq,
        core_name_freq=core_freq,
        addr_freq=addr_freq,
        name_token_freq=name_tok,
        addr_token_freq=addr_tok,
        total_names=total_names,
        total_addrs=total_addrs,
    )


# --------------------------------------------------------------------------- #
# Combined candidate space (S2 then S3) so candidate ids are a single integer
# --------------------------------------------------------------------------- #
@dataclass
class CandidateSpace:
    """A merged view over the non-reference sources, indexed by a global int id."""

    tables: Dict[str, SourceTable]          # {"S2": ..., "S3": ...}
    order: List[str]                        # e.g. ["S2", "S3"]
    offsets: Dict[str, int]
    total: int
    entity_ids: List[str]
    source_of: np.ndarray                   # uint8 source code per global id
    local_idx: np.ndarray                   # int32 local index within its source
    # flattened representation accessors (global-id aligned)
    name_light: List[str] = field(default_factory=list)
    name_core: List[str] = field(default_factory=list)
    name_heavy: List[str] = field(default_factory=list)
    name_tokens: List[TokenSeq] = field(default_factory=list)
    name_core_tokens: List[TokenSeq] = field(default_factory=list)
    addr_light: List[str] = field(default_factory=list)
    addr_heavy: List[str] = field(default_factory=list)
    addr_tokens: List[TokenSeq] = field(default_factory=list)
    postals: List[TokenSeq] = field(default_factory=list)
    house_no: List[str] = field(default_factory=list)
    country_norm: List[str] = field(default_factory=list)
    name_raw: List[str] = field(default_factory=list)
    addr_raw: List[str] = field(default_factory=list)


def build_candidate_space(tables: Dict[str, SourceTable], order: Sequence[str] = ("S2", "S3")) -> CandidateSpace:
    order = [s for s in order if s in tables and tables[s].n > 0]
    offsets: Dict[str, int] = {}
    ids: List[str] = []
    src_codes: List[int] = []
    local: List[int] = []
    off = 0
    for code, tag in enumerate(order):
        st = tables[tag]
        offsets[tag] = off
        ids.extend(st.entity_ids)
        src_codes.extend([code] * st.n)
        local.extend(range(st.n))
        off += st.n

    cs = CandidateSpace(
        tables={t: tables[t] for t in order},
        order=list(order),
        offsets=offsets,
        total=off,
        entity_ids=ids,
        source_of=np.asarray(src_codes, dtype=np.uint8),
        local_idx=np.asarray(local, dtype=np.int32),
    )
    for tag in order:
        st = tables[tag]
        cs.name_light.extend(st.name_light)
        cs.name_core.extend(st.name_core)
        cs.name_heavy.extend(st.name_heavy)
        cs.name_tokens.extend(st.name_tokens)
        cs.name_core_tokens.extend(st.name_core_tokens)
        cs.addr_light.extend(st.addr_light)
        cs.addr_heavy.extend(st.addr_heavy)
        cs.addr_tokens.extend(st.addr_tokens)
        cs.postals.extend(st.postals)
        cs.house_no.extend(st.house_no)
        cs.country_norm.extend(st.country_norm)
        cs.name_raw.extend(st.name_raw)
        cs.addr_raw.extend(st.addr_raw)
    return cs


def global_id(space: CandidateSpace, source: str, local: int) -> int:
    return space.offsets[source] + local
