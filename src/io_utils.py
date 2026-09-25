"""I/O layer: dataset discovery, schema introspection and memory-safe loading.

Design goals
------------
* Never hard-code a single absolute dataset path.
* Tolerate macOS metadata (``__MACOSX``, ``._*``, ``.DS_Store``) by ignoring it.
* Load only the columns we need, with explicit dtypes, and stream when large.
* Expose a single standard schema for every source:
      entity_id, business_name, business_address, country
"""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import polars as pl

from .utils import LOG, ensure_dir

# Canonical column names we normalise every file toward.
STANDARD_COLS = ["entity_id", "business_name", "business_address", "country"]

# Prefix -> source tag
SOURCE_PREFIXES = {"S1": "S1", "S2": "S2", "S3": "S3"}

_METADATA_NAMES = {".DS_Store"}
_METADATA_PREFIXES = ("._",)


def is_metadata_path(path: Path) -> bool:
    parts = set(path.parts)
    if "__MACOSX" in parts:
        return True
    name = path.name
    if name in _METADATA_NAMES:
        return True
    if name.startswith(_METADATA_PREFIXES):
        return True
    return False


@dataclass
class DatasetPaths:
    """Resolved locations of every file we care about."""

    root: Path
    train_gt: Optional[Path] = None
    train: Dict[str, Path] = field(default_factory=dict)  # {"S1": ..., "S2": ..., "S3": ...}
    test: Dict[str, Path] = field(default_factory=dict)
    validator: Optional[Path] = None
    doc_template: Optional[Path] = None
    readme: Optional[Path] = None

    def describe(self) -> str:
        lines = [f"data root : {self.root}"]
        lines.append(f"train gt  : {self.train_gt}")
        for k in ("S1", "S2", "S3"):
            lines.append(f"train {k}  : {self.train.get(k)}")
        for k in ("S1", "S2", "S3"):
            lines.append(f"test  {k}  : {self.test.get(k)}")
        lines.append(f"validator : {self.validator}")
        lines.append(f"doc tmpl  : {self.doc_template}")
        return "\n".join(lines)

    def is_complete(self) -> bool:
        if not self.train_gt or not self.train.get("S1"):
            return False
        if not self.test.get("S1"):
            return False
        return True


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #
def _find_first(candidates: Iterable[Path], pred) -> Optional[Path]:
    for c in sorted(candidates):
        if is_metadata_path(c):
            continue
        if pred(c):
            return c
    return None


def _collect_files(base: Path) -> List[Path]:
    out: List[Path] = []
    for dirpath, dirnames, filenames in os.walk(base):
        # prune macOS metadata directories early
        dirnames[:] = [d for d in dirnames if d != "__MACOSX" and not d.startswith("._")]
        for fn in filenames:
            p = Path(dirpath) / fn
            if not is_metadata_path(p):
                out.append(p)
    return out


def _search_roots() -> List[Path]:
    """Plausible locations to auto-detect the extracted resource."""
    roots: List[Path] = []
    env = os.environ.get("AMER_DATA_ROOT")
    if env:
        roots.append(Path(env).expanduser())
    roots += [
        Path.cwd(),
        Path.cwd() / "student_resource",
        Path.cwd().parent,
        Path(__file__).resolve().parents[1],  # project dir
        Path.home() / "Downloads",
        Path.home() / "Downloads" / "student_resource",
        Path("/data"),
        Path("/workspace"),
        Path("/mnt/data"),
        Path("/tmp"),
    ]
    return roots


def resolve_dataset(data_root: Optional[str] = None, search_depth: int = 5) -> DatasetPaths:
    """Locate the dataset. Accepts an explicit ``--data-root`` else auto-detects.

    Auto-detection searches plausible roots (env var, cwd, Downloads, /workspace,
    ...) recursively for the characteristic files.
    """
    bases: List[Path] = []
    if data_root:
        p = Path(data_root).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"--data-root does not exist: {p}")
        bases.append(p)
    else:
        for r in _search_roots():
            if r.exists() and r not in bases:
                bases.append(r)

    expected = {
        "train_ground_truth.tsv",
        "train_source1.tsv",
        "test_source1.tsv",
    }

    for base in bases:
        files = _collect_files(base)
        names = {f.name for f in files}
        if expected.issubset(names) or {"train_source1.tsv", "test_source1.tsv"} <= names:
            return _build_paths(base, files)

    # deeper explicit recursive search from the given / common roots
    if data_root:
        base = Path(data_root).expanduser().resolve()
        files = _collect_files(base)
        if files:
            return _build_paths(base, files)

    raise FileNotFoundError(
        "Could not locate the dataset. Pass --data-root pointing at the directory "
        "that contains dataset/train/train_source1.tsv and dataset/test/test_source1.tsv."
    )


def _build_paths(base: Path, files: Sequence[Path]) -> DatasetPaths:
    by_name: Dict[str, Path] = {}
    for f in sorted(files):
        by_name.setdefault(f.name, f)

    def get(name: str) -> Optional[Path]:
        return by_name.get(name)

    paths = DatasetPaths(root=base)
    paths.train_gt = get("train_ground_truth.tsv")
    for src in ("1", "2", "3"):
        t = get(f"train_source{src}.tsv")
        if t:
            paths.train[f"S{src}"] = t
        te = get(f"test_source{src}.tsv")
        if te:
            paths.test[f"S{src}"] = te
    paths.validator = get("validate_submission.py")
    paths.doc_template = get("Documentation_template.md")
    paths.readme = get("README.md")
    return paths


# --------------------------------------------------------------------------- #
# Schema introspection
# --------------------------------------------------------------------------- #
def sniff_header(path: Path, encoding: str = "utf-8") -> List[str]:
    """Read only the first line to obtain the header."""
    with open(path, "r", encoding=encoding, errors="replace") as fh:
        first = fh.readline().rstrip("\n").rstrip("\r")
    delim = "\t" if "\t" in first else ("," if "," in first else None)
    if delim is None:
        return [first]
    return [c.strip().lstrip("\ufeff") for c in first.split(delim)]


def _resolve_columns(header: Sequence[str]) -> Dict[str, str]:
    """Map our canonical names -> actual header names, tolerantly."""
    lower = {h.lower().strip(): h for h in header}

    def pick(*aliases: str) -> Optional[str]:
        for a in aliases:
            if a in lower:
                return lower[a]
        # substring fallback
        for h in header:
            hl = h.lower()
            for a in aliases:
                if a in hl:
                    return h
        return None

    mapping = {
        "entity_id": pick("entity_id", "entityid", "id", "entity"),
        "business_name": pick("business_name", "name", "businessname", "company_name"),
        "business_address": pick("business_address", "address", "addr"),
        "country": pick("country", "nation"),
    }
    return {k: v for k, v in mapping.items() if v is not None}


def count_rows_fast(path: Path) -> int:
    """Count data rows cheaply without parsing (minus the header)."""
    n = 0
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for _ in fh:
            n += 1
    return max(0, n - 1)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_source(
    path: Path,
    lazy: bool = False,
    n_rows: Optional[int] = None,
) -> "pl.DataFrame | pl.LazyFrame":
    """Load one source TSV into the standard schema.

    Uses ``pl.read_csv`` with a tab separator and string dtypes (entity strings
    are identifiers; we do not want accidental numeric coercion). ``lazy=True``
    returns a LazyFrame for streaming aggregation.
    """
    header = sniff_header(path)
    cols = _resolve_columns(header)
    if "entity_id" not in cols:
        raise ValueError(f"No entity_id-like column found in {path}: {header}")

    rename = {v: k for k, v in cols.items()}
    read = pl.scan_csv if lazy else pl.read_csv
    kwargs = dict(
        separator="\t",
        has_header=True,
        infer_schema_length=0,  # read everything as strings first
        quote_char=None,        # TSVs are not quoted; avoids surprises
        truncate_ragged_lines=True,
        encoding="utf8-lossy",
        n_rows=n_rows,
    )
    try:
        df = read(path, **kwargs)
    except Exception as exc:  # pragma: no cover - fallback for odd files
        LOG.warning("scan_csv failed (%s); falling back to csv module for %s", exc, path)
        df = _load_with_csv(path, lazy=lazy)

    present = [c for c in rename if c in df.collect_schema().names()]
    df = df.rename({rename[c]: c for c in present})
    if lazy:
        have = df.collect_schema().names()
        for c in STANDARD_COLS:
            if c not in have:
                df = df.with_columns(pl.lit(None).cast(pl.Utf8).alias(c))
        return df.select(STANDARD_COLS)
    have = df.columns
    for c in STANDARD_COLS:
        if c not in have:
            df = df.with_columns(pl.lit(None).cast(pl.Utf8).alias(c))
    return df.select(STANDARD_COLS)


def _load_with_csv(path: Path, lazy: bool = False):
    header = sniff_header(path)
    idx = {h: i for i, h in enumerate(header)}
    rows: Dict[str, List[Optional[str]]] = {c: [] for c in STANDARD_COLS}
    cols = _resolve_columns(header)
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        rdr = csv.reader(fh, delimiter="\t")
        next(rdr, None)
        for row in rdr:
            for canon, actual in cols.items():
                i = idx.get(actual)
                rows[canon].append(row[i] if i is not None and i < len(row) else None)
    data = {k: pl.Series(k, v, dtype=pl.Utf8) for k, v in rows.items()}
    for c in STANDARD_COLS:
        data.setdefault(c, pl.Series(c, [None] * len(next(iter(rows.values()))), dtype=pl.Utf8))
    df = pl.DataFrame(data)
    return df.lazy() if lazy else df


# --------------------------------------------------------------------------- #
# Ground truth
# --------------------------------------------------------------------------- #
def load_ground_truth(path: Path) -> "pl.DataFrame":
    """Load ground truth into ``source1_entity_id`` + ``matched_entity_ids`` (raw str)."""
    header = sniff_header(path)
    lower = {h.lower(): h for h in header}

    id_col = None
    for cand in ("source1_entity_id", "entity_id", "source_1_entity_id", "s1_entity_id"):
        if cand in lower:
            id_col = lower[cand]
            break
    if id_col is None:
        # first column
        id_col = header[0]

    match_col = None
    for cand in ("matched_entity_ids", "matched_ids", "matches", "matched_entity_id"):
        if cand in lower:
            match_col = lower[cand]
            break
    if match_col is None and len(header) > 1:
        match_col = header[1]

    df = pl.read_csv(
        path,
        separator="\t",
        has_header=True,
        infer_schema_length=0,
        quote_char=None,
        truncate_ragged_lines=True,
        encoding="utf8-lossy",
    )
    df = df.rename({id_col: "source1_entity_id"})
    if match_col and match_col in df.columns:
        df = df.rename({match_col: "matched_entity_ids"})
    else:
        df = df.with_columns(pl.lit(None).cast(pl.Utf8).alias("matched_entity_ids"))
    return df.select(["source1_entity_id", "matched_entity_ids"])


def parse_match_list(raw: Optional[str]) -> List[str]:
    """Parse a comma-separated match field into a clean, de-duplicated list."""
    if raw is None:
        return []
    s = str(raw).strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return []
    parts = [p.strip() for p in s.replace(";", ",").split(",")]
    seen, out = set(), []
    for p in parts:
        if p and p not in seen:
            seen.add(p)
            out.append(p)
    return out


def write_tsv(df: "pl.DataFrame", path: Path) -> None:
    ensure_dir(Path(path).parent)
    df.write_csv(path, separator="\t", quote_style="never", null_value="")
