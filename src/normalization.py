"""Normalization and text representations.

We deliberately keep *multiple* representations rather than collapsing to one
string, because different features need different views:

  original        -> error analysis / interpretability only
  light_normalized-> casefold, whitespace, punctuation cleanup, & -> and
  heavy_normalized-> light + legal-suffix / abbreviation canonicalisation
  core            -> heavy with legal suffixes *removed* (business "core" name)
  tokens          -> token list of light_normalized

Addresses additionally yield numeric tokens, house-number, and postal candidates.
No external geocoding or data is ever used.
"""
from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------- #
# Character-level cleanup
# --------------------------------------------------------------------------- #
_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_WS_RE = re.compile(r"\s+", re.UNICODE)
_MULTI_WS = _WS_RE

# Common unicode punctuation/dash/quotes -> ascii equivalents
_UNICODE_MAP = {
    "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
    "\u2013": "-", "\u2014": "-", "\u2212": "-", "\u00a0": " ",
    "\u200b": "", "\ufeff": "", "\u2026": "...",
}

# & and + are common in Indian business names
_AND_PATTERNS = [
    (re.compile(r"\s*&\s*"), " and "),
    (re.compile(r"\s*\+\s*"), " and "),
    (re.compile(r"\s+@\s+"), " at "),
]


def strip_accents(text: str) -> str:
    """NFKD-decompose and drop combining marks (a safe transliteration)."""
    if not text:
        return ""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def unicode_fold(text: Optional[str]) -> str:
    """Unicode-normalise, map fancy punctuation, strip accents, casefold."""
    if text is None:
        return ""
    s = str(text)
    for k, v in _UNICODE_MAP.items():
        if k in s:
            s = s.replace(k, v)
    s = unicodedata.normalize("NFKC", s)
    s = strip_accents(s)
    return s.casefold()


def clean_text(text: Optional[str]) -> str:
    """Light normalization: fold -> punctuation to spaces -> collapse whitespace."""
    s = unicode_fold(text)
    for pat, rep in _AND_PATTERNS:
        s = pat.sub(rep, s)
    s = _PUNCT_RE.sub(" ", s)
    s = _MULTI_WS.sub(" ", s).strip()
    return s


# --------------------------------------------------------------------------- #
# Business-name abbreviation / legal-suffix canonicalisation
# --------------------------------------------------------------------------- #
_LEGAL_TOKENS = {
    "incorporated", "inc", "corporation", "corp", "corporate",
    "company", "co", "limited", "ltd", "llc", "llp", "lp", "plc",
    "private", "pvt", "pvtltd", "pvtltd", "pte", "srl", "gmbh", "sa",
    "ag", "bv", "nv", "sas", "sarl", "spa", "kk", "oy", "ab", "as",
    "pty", "trust", "foundation", "enterprises", "enterprise",
    "international", "intl", "group", "holdings", "holding",
}

# Multi-word abbreviations must be applied before token-level legal removal.
_MULTIWORD_ABBREV = [
    (re.compile(r"\bprivate\s+limited\b"), "pvt ltd"),
    (re.compile(r"\bpvt\.?\s*ltd\.?\b"), "pvt ltd"),
    # only expand a *bare* pvt that is not already followed by ltd
    (re.compile(r"\bpvt\b(?!\s*ltd)"), "pvt ltd"),
    (re.compile(r"\bsociete\s+anonyme\b"), "sa"),
    (re.compile(r"\bco\s+limited\b"), "co ltd"),
]

_TOKEN_CANON = {
    "incorporated": "inc", "corporation": "corp", "corporate": "corp",
    "company": "co", "limited": "ltd", "private": "pvt",
    "international": "intl", "holdings": "holding",
    "and": "and",
}

# Road-type abbreviations for addresses
_ADDR_CANON = {
    "road": "rd", "street": "st", "avenue": "ave", "avenu": "ave",
    "boulevard": "blvd", "drive": "dr", "lane": "ln", "highway": "hwy",
    "parkway": "pkwy", "place": "pl", "court": "ct", "circle": "cir",
    "square": "sq", "terrace": "ter", "nagar": "nagar",
    "cross": "cross", "main": "main", "layout": "layout",
    "sector": "sector", "block": "block", "phase": "phase",
    "floor": "flr", "building": "bldg", "apartment": "apt",
    "suite": "ste", "number": "no", "opposite": "opp",
    "near": "near", "post": "po", "district": "dist",
    "north": "n", "south": "s", "east": "e", "west": "w",
    "northern": "n", "southern": "s", "eastern": "e", "western": "w",
}

# Country canonicalisation (open-set friendly)
_COUNTRY_CANON = {
    "us": "united states", "usa": "united states", "u s a": "united states",
    "u s": "united states", "united states of america": "united states",
    "america": "united states",
    "in": "india", "ind": "india", "bharat": "india",
    "fr": "france", "fra": "france", "french republic": "france",
    "uk": "united kingdom", "gb": "united kingdom", "great britain": "united kingdom",
    "uae": "united arab emirates",
    "ca": "canada", "cn": "china", "de": "germany", "au": "australia",
    "sg": "singapore", "ae": "united arab emirates",
}

_NUM_RE = re.compile(r"\d+")
_POSTAL_RE = re.compile(r"\b(\d{5,6}(?:[- ]\d{3,4})?)\b")
_ALNUM_RE = re.compile(r"[a-z0-9]+")

_STOP_TOKENS = {"the", "of", "and", "at", "for", "to", "in", "a", "an", "on"}


# --------------------------------------------------------------------------- #
# Name representations
# --------------------------------------------------------------------------- #
def light_normalize_name(text: Optional[str]) -> str:
    return clean_text(text)


def heavy_normalize_name(text: Optional[str]) -> str:
    s = clean_text(text)
    for pat, rep in _MULTIWORD_ABBREV:
        s = pat.sub(rep, s)
    toks = [t for t in s.split() if t]
    out = []
    for t in toks:
        t2 = _TOKEN_CANON.get(t, t)
        out.append(t2)
    return " ".join(out)


def core_name(text: Optional[str]) -> str:
    """Heavy-normalized name with legal/business-type tokens removed."""
    s = heavy_normalize_name(text)
    toks = [t for t in s.split() if t and t not in _LEGAL_TOKENS]
    # de-duplicate consecutive duplicate tokens only
    dedup: List[str] = []
    for t in toks:
        if not dedup or dedup[-1] != t:
            dedup.append(t)
    return " ".join(dedup)


def name_tokens(text: Optional[str]) -> List[str]:
    s = light_normalize_name(text)
    return [t for t in s.split() if t]


def name_core_tokens(text: Optional[str]) -> List[str]:
    return [t for t in core_name(text).split() if t]


def drop_stopwords(tokens: Sequence[str]) -> List[str]:
    return [t for t in tokens if t not in _STOP_TOKENS]


# --------------------------------------------------------------------------- #
# Address representations
# --------------------------------------------------------------------------- #
def normalize_address_light(text: Optional[str]) -> str:
    return clean_text(text)


def normalize_address_heavy(text: Optional[str]) -> str:
    s = clean_text(text)
    toks = [t for t in s.split() if t]
    out = []
    for t in toks:
        if t in _ADDR_CANON:
            out.append(_ADDR_CANON[t])
        elif t.isdigit():
            out.append(str(int(t)))  # strip leading zeros
        else:
            out.append(t)
    # collapse consecutive duplicates
    dedup: List[str] = []
    for t in out:
        if not dedup or dedup[-1] != t:
            dedup.append(t)
    return " ".join(dedup)


def address_tokens(text: Optional[str]) -> List[str]:
    return [t for t in normalize_address_light(text).split() if t]


def address_numbers(text: Optional[str]) -> List[str]:
    return _NUM_RE.findall(text or "")


def address_alnum_tokens(text: Optional[str]) -> List[str]:
    return _ALNUM_RE.findall(unicode_fold(text))


def postal_candidates(text: Optional[str]) -> List[str]:
    """Extract postal/PIN-like numeric sequences (US 5, India 6, US ZIP+4)."""
    nums = _NUM_RE.findall(text or "")
    out = [n for n in nums if 5 <= len(n) <= 6]
    # US ZIP+4
    out += [n for n in _POSTAL_RE.findall(text or "")]
    # also expose long numeric tokens truncated to 6 as weak candidates
    for n in nums:
        if len(n) > 6 and len(n) <= 9:
            out.append(n[:6])
    seen, res = set(), []
    for n in out:
        if n not in seen:
            seen.add(n)
            res.append(n)
    return res


def house_number(text: Optional[str]) -> str:
    """First numeric token of an address, normalized (leading zeros stripped)."""
    nums = address_numbers(text)
    if not nums:
        return ""
    return str(int(nums[0]))


# --------------------------------------------------------------------------- #
# Country
# --------------------------------------------------------------------------- #
def normalize_country(text: Optional[str]) -> str:
    if text is None:
        return ""
    s = clean_text(text)
    s = _COUNTRY_CANON.get(s, s)
    return s


# --------------------------------------------------------------------------- #
# Batched representation builder
# --------------------------------------------------------------------------- #
class TextRepresenter:
    """Produce the full multi-representation dict for one record."""

    __slots__ = ()

    @staticmethod
    def name_reprs(raw: Optional[str]) -> Dict[str, object]:
        light = light_normalize_name(raw)
        heavy = heavy_normalize_name(raw)
        core = core_name(raw)
        toks = name_tokens(raw)
        return {
            "business_name_light_normalized": light,
            "business_name_heavy_normalized": heavy,
            "business_name_core": core,
            "business_name_tokens": toks,
            "business_name_core_tokens": name_core_tokens(raw),
        }

    @staticmethod
    def address_reprs(raw: Optional[str]) -> Dict[str, object]:
        return {
            "business_address_normalized": normalize_address_light(raw),
            "business_address_heavy": normalize_address_heavy(raw),
            "business_address_tokens": address_tokens(raw),
            "address_numbers": address_numbers(raw),
            "address_alnum_tokens": address_alnum_tokens(raw),
            "postal_candidates": postal_candidates(raw),
            "house_number": house_number(raw),
        }

    @staticmethod
    def country_repr(raw: Optional[str]) -> Dict[str, object]:
        return {
            "country_normalized": normalize_country(raw),
            "country_present": bool(normalize_country(raw)),
        }
