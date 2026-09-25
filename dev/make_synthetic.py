#!/usr/bin/env python3
"""Synthetic dataset generator for SELF-TESTING the pipeline only.

The real challenge data is NOT in this sandbox (it lives on the user's Windows
machine). To verify the pipeline actually runs end-to-end, produces the two TSVs
and passes the validator, this script fabricates a dataset with the *same schema*
and analogous difficulty:

  * S1 is the deduplicated reference; S2 and S3 are messy variants
  * entities may have zero (singleton), one, or many matches
  * names are perturbed (case, punctuation, legal suffixes, abbreviations,
    transliteration, token reordering, truncation)
  * addresses are perturbed (road/rd, st/street, PIN formatting, house numbers)
  * train holds US + India; test additionally holds France (unseen country)

Any metric produced from this data is for plumbing verification ONLY and is not a
claim about real leaderboard performance.
"""
from __future__ import annotations

import argparse
import random
import unicodedata
from pathlib import Path

CITIES = {
    "united states": ["Springfield", "Riverside", "Fairview", "Georgetown", "Madison", "Clinton"],
    "india": ["Chennai", "Mumbai", "Bengaluru", "Pune", "Hyderabad", "Kochi"],
    "france": ["Lyon", "Nantes", "Lille", "Toulouse", "Bordeaux", "Rennes"],
}
STREETS = ["Main", "Oak", "Park", "Lake", "Hill", "Church", "Market", "Gandhi", "Nehru", "Rue Victor Hugo", "Rue de la Paix"]
SUFFIXES = ["", "Inc", "Corp", "Ltd", "LLC", "Pvt Ltd", "Company", "International"]
TYPES = [
    "Electronics", "Traders", "Enterprises", "Solutions", "Industries", "Trading Co",
    "General Stores", "Textiles", "Motors", "Foods", "Pharma", "Logistics",
]
FIRST = ["Sharma", "Patel", "Kumar", "Rao", "Iyer", "Nair", "Smith", "Johnson", "Brown", "Miller", "Dubois", "Martin", "Bernard"]
STEM = ["Alpha", "Zenith", "Prime", "Sunrise", "Golden", "Blue", "Metro", "Star", "Global", "Union", "Green", "Royal"]


def _accent(s: str) -> str:
    table = {"e": "é", "a": "à", "c": "ç", "u": "û", "o": "ô"}
    out = []
    for i, ch in enumerate(s):
        if i % 4 == 2 and ch in table:
            out.append(table[ch])
        else:
            out.append(ch)
    return "".join(out)


def make_name(rng: random.Random, country: str) -> str:
    style = rng.random()
    if style < 0.45:
        n = f"{rng.choice(STEM)} {rng.choice(TYPES)}"
    elif style < 0.75:
        n = f"{rng.choice(FIRST)} {rng.choice(TYPES)}"
    else:
        n = f"{rng.choice(FIRST)} {rng.choice(STEM)} {rng.choice(TYPES)}"
    suf = rng.choice(SUFFIXES)
    if suf:
        n = f"{n} {suf}"
    if country == "france" and rng.random() < 0.3:
        n = _accent(n)
    return n


def make_addr(rng: random.Random, country: str) -> str:
    num = rng.randint(1, 999)
    street = rng.choice(STREETS)
    city = rng.choice(CITIES[country])
    if country == "india":
        pin = rng.randint(100000, 999999)
        return f"{num} {street} Street, {city} {pin}, {country.title()}"
    if country == "france":
        return f"{num} {street}, {city}, France"
    return f"{num} {street} Ave, {city}, {country.title()}"


def perturb_name(rng: random.Random, name: str) -> str:
    ops = rng.sample(
        ["lower", "upper", "punct", "abbrev", "drop_suffix", "reorder", "truncate", "dup_token", "and"],
        k=rng.randint(1, 3),
    )
    out = name
    for op in ops:
        if op == "lower":
            out = out.lower()
        elif op == "upper":
            out = out.upper()
        elif op == "punct":
            out = out.replace(" ", rng.choice([" ", ". ", " - ", ", "]), 1)
        elif op == "abbrev":
            out = out.replace("Limited", "Ltd").replace("Private", "Pvt").replace("Company", "Co")
        elif op == "drop_suffix":
            for s in ("Inc", "Corp", "Ltd", "LLC", "Pvt Ltd", "Company", "International"):
                out = out.replace(" " + s, "")
        elif op == "reorder":
            parts = out.split()
            if len(parts) >= 3:
                parts[0], parts[1] = parts[1], parts[0]
                out = " ".join(parts)
        elif op == "truncate":
            out = out[: max(4, len(out) - rng.randint(1, 4))]
        elif op == "dup_token":
            parts = out.split()
            if parts:
                parts.insert(0, parts[0])
            out = " ".join(parts)
        elif op == "and":
            out = out.replace("&", "and").replace("and", "&")
    return out.strip()


def perturb_addr(rng: random.Random, addr: str) -> str:
    out = addr
    subs = [
        ("Street", rng.choice(["St", "St.", "street"])),
        ("Ave", rng.choice(["Avenue", "Ave.", "ave"])),
        (",", rng.choice([",", " ,", ""])),
        ("  ", " "),
    ]
    for a, b in subs:
        if rng.random() < 0.5:
            out = out.replace(a, b)
    if rng.random() < 0.25:
        parts = out.split()
        if parts:
            parts[0] = str(int(parts[0]) if parts[0].isdigit() else rng.randint(1, 999))
            out = " ".join(parts)
    return out


def generate(out_dir: Path, n_train_s1: int, n_test_s1: int, seed: int = 13) -> None:
    rng = random.Random(seed)
    (out_dir / "dataset" / "train").mkdir(parents=True, exist_ok=True)
    (out_dir / "dataset" / "test").mkdir(parents=True, exist_ok=True)
    (out_dir / "utils").mkdir(parents=True, exist_ok=True)

    def write_split(split: str, n_s1: int, countries: list):
        s1_rows = []
        s2 = {}
        s3 = {}
        gt = []
        si = 0
        s2c = s3c = 0
        for _ in range(n_s1):
            si += 1
            s1_id = f"S1-{si:05d}"
            country = rng.choice(countries)
            name = make_name(rng, country)
            addr = make_addr(rng, country)
            s1_rows.append((s1_id, name, addr, country.title()))
            matches = []
            # how many true matches this entity has (sometimes none)
            r = rng.random()
            if r < 0.30:
                n_match = 0
            elif r < 0.75:
                n_match = 1
            else:
                n_match = rng.randint(2, 3)
            for _ in range(n_match):
                for src, store, pref in (("S2", s2, "S2"), ("S3", s3, "S3")):
                    if rng.random() < 0.5:
                        continue
                    if src == "S2":
                        s2c += 1
                        mid = f"S2-{s2c:06d}"
                    else:
                        s3c += 1
                        mid = f"S3-{s3c:06d}"
                    store[mid] = (mid, perturb_name(rng, name), perturb_addr(rng, addr), country.title())
                    matches.append(mid)
            gt.append((s1_id, ",".join(matches)))
            # decoys: same country, similar but distinct businesses (hard negatives)
            for _ in range(rng.randint(0, 2)):
                dname = make_name(rng, country)
                daddr = make_addr(rng, country)
                s2c += 1
                s2[f"S2-{s2c:06d}"] = (f"S2-{s2c:06d}", dname, daddr, country.title())

        prefix = "train" if split == "train" else "test"
        _write_tsv(out_dir / "dataset" / split / f"{prefix}_source1.tsv",
                   ["entity_id", "business_name", "business_address", "country"], s1_rows)
        _write_tsv(out_dir / "dataset" / split / f"{prefix}_source2.tsv",
                   ["entity_id", "business_name", "business_address", "country"], list(s2.values()))
        _write_tsv(out_dir / "dataset" / split / f"{prefix}_source3.tsv",
                   ["entity_id", "business_name", "business_address", "country"], list(s3.values()))
        if split == "train":
            _write_tsv(out_dir / "dataset" / split / "train_ground_truth.tsv",
                       ["source1_entity_id", "matched_entity_ids"], gt)

    write_split("train", n_train_s1, ["united states", "india"])
    write_split("test", n_test_s1, ["united states", "india", "france"])

    # a stand-in official validator with the same CLI contract
    (out_dir / "utils" / "validate_submission.py").write_text(VALIDATOR_SRC, encoding="utf-8")
    (out_dir / "Documentation_template.md").write_text("# Documentation\n\n(fill me)\n", encoding="utf-8")
    (out_dir / "README.md").write_text("# Synthetic student_resource\n", encoding="utf-8")
    print(f"synthetic dataset written to {out_dir}")


def _write_tsv(path: Path, header, rows) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\t".join(header) + "\n")
        for r in rows:
            fh.write("\t".join(str(x) for x in r) + "\n")


VALIDATOR_SRC = '''#!/usr/bin/env python3
"""Stand-in official validator (same CLI contract as the challenge one)."""
import argparse, sys
from pathlib import Path

def read_pairs(p):
    rows = {}
    for ln in Path(p).read_text(encoding="utf-8").splitlines()[1:]:
        if not ln:
            continue
        parts = ln.split("\\t")
        if len(parts) != 2:
            rows.setdefault("__malformed__", 0)
            rows["__malformed__"] += 1
            continue
        rows[parts[0]] = parts[1]
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matching", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--test-dir", required=True)
    a = ap.parse_args()
    test_dir = Path(a.test_dir)
    s1 = Path(test_dir) / "test_source1.tsv"
    s2 = Path(test_dir) / "test_source2.tsv"
    s3 = Path(test_dir) / "test_source3.tsv"
    ids = lambda p: {ln.split("\\t")[0] for ln in p.read_text(encoding="utf-8").splitlines()[1:] if ln}
    s1_ids, s2_ids, s3_ids = ids(s1), ids(s2), ids(s3)
    match = read_pairs(a.matching)
    cand = read_pairs(a.candidate)
    errs = []
    if len(match) != len(s1_ids):
        errs.append(f"matching rows {len(match)} != S1 {len(s1_ids)}")
    cand_sets = {k: set(v.split(",")) - {""} for k, v in cand.items()}
    for sid, matched in match.items():
        if sid not in s1_ids:
            errs.append(f"unknown S1 {sid}")
        for mid in (set(matched.split(",")) - {""}):
            if not (mid in s2_ids or mid in s3_ids):
                errs.append(f"match {mid} not in test")
            if mid not in cand_sets.get(sid, set()):
                errs.append(f"match {mid} not in candidates of {sid}")
    missing = s1_ids - set(match)
    if missing:
        errs.append(f"{len(missing)} S1 entities missing")
    if errs:
        print("FAIL")
        for e in errs[:20]:
            print(" ", e)
        sys.exit(1)
    print("PASS")
    print(f"matching rows: {len(match)}; test S1: {len(s1_ids)}; S2: {len(s2_ids)}; S3: {len(s3_ids)}")
    sys.exit(0)

if __name__ == "__main__":
    main()
'''


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dev/synthetic/student_resource")
    ap.add_argument("--n-train", type=int, default=1500)
    ap.add_argument("--n-test", type=int, default=500)
    ap.add_argument("--seed", type=int, default=13)
    a = ap.parse_args()
    generate(Path(a.out), a.n_train, a.n_test, a.seed)
