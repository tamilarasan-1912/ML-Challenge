"""Dataset profiling: real, measured statistics only.

All numbers come from the actual files. Large files are counted with cheap
streaming (row counts) and aggregated with polars lazy expressions so that we
never materialise a multi-GB frame unnecessarily.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import polars as pl

from . import io_utils as io
from .normalization import (
    address_alnum_tokens,
    address_numbers,
    name_tokens,
    postal_candidates,
    normalize_country,
)
from .utils import LOG, ensure_dir, write_json


def _field_stats(series: "pl.Series") -> dict:
    s = series.cast(pl.Utf8).fill_null("")
    n = len(s)
    empty = int((s.str.strip_chars() == "").sum())
    vals = s.to_list()
    lens = np_len(vals)
    import numpy as np

    arr = np.asarray(lens, dtype=np.int64)
    return {
        "n": n,
        "null_count": int(series.null_count()),
        "empty_or_blank": empty,
        "unique": int(s.n_unique()),
        "min_len": int(arr.min()) if n else 0,
        "max_len": int(arr.max()) if n else 0,
        "mean_len": float(arr.mean()) if n else 0.0,
        "median_len": float(np.median(arr)) if n else 0.0,
        "p95_len": float(np.percentile(arr, 95)) if n else 0.0,
    }


def np_len(vals: List[str]):
    return [len(v) for v in vals]


def _token_stats(series: "pl.Series", kind: str) -> dict:
    s = series.cast(pl.Utf8).fill_null("")
    counts: Counter = Counter()
    token_counts: Counter = Counter()
    total_tokens = 0
    numeric_tokens = 0
    postal_hits = 0
    ntoks = []
    for v in s.to_list():
        counts[v.strip().lower()] += 1
        if kind == "name":
            toks = name_tokens(v)
        else:
            toks = [t for t in (v or "").lower().split() if t]
            if postal_candidates(v):
                postal_hits += 1
            numeric_tokens += len(address_numbers(v))
        ntoks.append(len(toks))
        total_tokens += len(toks)
        for t in toks:
            token_counts[t] += 1

    import numpy as np

    arr = np.asarray(ntoks, dtype=np.int64)
    res = {
        "total_tokens": int(total_tokens),
        "mean_tokens": float(arr.mean()) if len(arr) else 0.0,
        "max_tokens": int(arr.max()) if len(arr) else 0,
        "top_tokens": token_counts.most_common(25),
        "rare_tokens": sorted(token_counts.items(), key=lambda kv: kv[1])[:25],
        "vocab_size": len(token_counts),
    }
    if kind == "address":
        res["numeric_token_total"] = int(numeric_tokens)
        res["records_with_postal_like"] = int(postal_hits)
    return res


def profile_source(path: Path, tag: str, sample_full: bool = True) -> dict:
    LOG.info("profiling %s (%s)", tag, path)
    header = io.sniff_header(path)
    n_rows = io.count_rows_fast(path)
    info = {
        "path": str(path),
        "file_size_bytes": path.stat().st_size,
        "header": header,
        "n_rows": n_rows,
        "columns": {},
    }

    # load relevant columns (string dtype) for detailed stats
    df = io.load_source(path)
    rename_cols = {
        "entity_id": "entity_id",
        "business_name": "business_name",
        "business_address": "business_address",
        "country": "country",
    }
    for col in ("entity_id", "business_name", "business_address", "country"):
        if col in df.columns:
            sub = df[col]
            if col == "entity_id":
                vals = sub.cast(pl.Utf8).fill_null("").to_list()
                pref = Counter(v.split("-")[0] if "-" in v else "(none)" for v in vals)
                dups = n_rows - len(set(vals))
                info["columns"][col] = {
                    **_field_stats(sub),
                    "duplicate_ids": int(dups),
                    "prefix_distribution": dict(pref.most_common(10)),
                }
            elif col == "business_name":
                info["columns"][col] = {**_field_stats(sub), **_token_stats(sub, "name")}
            elif col == "business_address":
                info["columns"][col] = {**_field_stats(sub), **_token_stats(sub, "address")}
            elif col == "country":
                vals = sub.cast(pl.Utf8).fill_null("").to_list()
                norm = Counter(normalize_country(v) for v in vals)
                info["columns"][col] = {
                    **_field_stats(sub),
                    "distribution": dict(norm.most_common(30)),
                }

    # missing-field combinations
    idf = df.with_columns(
        [
            (pl.col("business_name").cast(pl.Utf8).fill_null("").str.strip_chars() == "").alias("name_missing"),
            (pl.col("business_address").cast(pl.Utf8).fill_null("").str.strip_chars() == "").alias("addr_missing"),
            (pl.col("country").cast(pl.Utf8).fill_null("").str.strip_chars() == "").alias("country_missing"),
        ]
    )
    combo = idf.group_by(["name_missing", "addr_missing", "country_missing"]).len().to_dicts()
    info["missing_field_combinations"] = combo

    # unicode / punctuation presence
    names = df["business_name"].cast(pl.Utf8).fill_null("").to_list()
    unicode_hits = sum(1 for v in names if any(ord(c) > 127 for c in v))
    punct_hits = sum(1 for v in names if any((not c.isalnum()) and (not c.isspace()) for c in v))
    info["name_unicode_records"] = unicode_hits
    info["name_with_punctuation_records"] = punct_hits

    if isinstance(df, pl.DataFrame):
        del df
    return info


def profile_ground_truth(gt_path: Path) -> dict:
    LOG.info("profiling ground truth %s", gt_path)
    gt = io.load_ground_truth(gt_path)
    total = len(gt)
    counts = Counter()
    by_src = Counter()
    s2_pos = 0
    s3_pos = 0
    empty = 0
    for raw in gt["matched_entity_ids"].to_list():
        ids = io.parse_match_list(raw)
        counts[len(ids)] += 1
        if not ids:
            empty += 1
        for mid in ids:
            if mid.startswith("S2-"):
                s2_pos += 1
                by_src["S2"] += 1
            elif mid.startswith("S3-"):
                s3_pos += 1
                by_src["S3"] += 1
            else:
                by_src["other"] += 1
    return {
        "path": str(gt_path),
        "n_rows": total,
        "n_unique_s1": int(gt["source1_entity_id"].n_unique()),
        "duplicate_s1_rows": int(total - gt["source1_entity_id"].n_unique()),
        "match_count_distribution": {str(k): int(v) for k, v in sorted(counts.items())},
        "n_singletons": int(counts.get(0, 0)),
        "n_with_matches": int(total - counts.get(0, 0)),
        "n_multi_match_entities": int(sum(v for k, v in counts.items() if k >= 2)),
        "positive_match_ids_total": int(sum(k * v for k, v in counts.items())),
        "positive_by_source": dict(by_src),
        "s2_positive_count": s2_pos,
        "s3_positive_count": s3_pos,
    }


def build_profile(data_root: str, out_dir: str, sample_rows: Optional[int] = None) -> dict:
    import numpy as np  # noqa: F401  (used indirectly by helpers)

    paths = io.resolve_dataset(data_root)
    LOG.info("resolved dataset:\n%s", paths.describe())
    profile = {
        "data_root": str(paths.root),
        "resolved": {
            "train_gt": str(paths.train_gt) if paths.train_gt else None,
            "train": {k: str(v) for k, v in paths.train.items()},
            "test": {k: str(v) for k, v in paths.test.items()},
            "validator": str(paths.validator) if paths.validator else None,
        },
        "train": {},
        "test": {},
    }
    for tag, p in paths.train.items():
        profile["train"][tag] = profile_source(p, f"train_{tag}")
    for tag, p in paths.test.items():
        profile["test"][tag] = profile_source(p, f"test_{tag}")
    if paths.train_gt:
        profile["ground_truth"] = profile_ground_truth(paths.train_gt)

    out = Path(out_dir)
    ensure_dir(out)
    write_json(out / "data_profile.json", profile)
    (out / "data_profile.md").write_text(render_profile_md(profile), encoding="utf-8")
    LOG.info("wrote %s and %s", out / "data_profile.json", out / "data_profile.md")
    return profile


def render_profile_md(profile: dict) -> str:
    L: List[str] = ["# Data Profile", ""]
    L.append(f"Data root: `{profile['data_root']}`")
    L.append("")
    L.append("## Resolved files")
    for k, v in profile["resolved"]["train"].items():
        L.append(f"- train {k}: `{v}`")
    for k, v in profile["resolved"]["test"].items():
        L.append(f"- test {k}: `{v}`")
    L.append("")

    for split in ("train", "test"):
        L.append(f"## {split.capitalize()} sources")
        for tag, info in profile[split].items():
            L.append(f"### {split} {tag}")
            L.append(f"- file: `{info['path']}`")
            L.append(f"- size: {info['file_size_bytes']:,} bytes")
            L.append(f"- rows: {info['n_rows']:,}")
            L.append(f"- header: {info['header']}")
            for col, cs in info["columns"].items():
                L.append(f"- column `{col}`:")
                for k, v in cs.items():
                    if k in ("top_tokens", "rare_tokens"):
                        v = ", ".join(f"{t}:{c}" for t, c in v[:12])
                    L.append(f"    - {k}: {v}")
            L.append(f"- missing field combinations: {info.get('missing_field_combinations')}")
            L.append(f"- name records with non-ASCII: {info.get('name_unicode_records')}")
            L.append(f"- name records with punctuation: {info.get('name_with_punctuation_records')}")
            L.append("")

    gt = profile.get("ground_truth")
    if gt:
        L.append("## Ground truth")
        for k, v in gt.items():
            L.append(f"- {k}: {v}")
        L.append("")
    return "\n".join(L)
