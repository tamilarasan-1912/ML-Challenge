"""Ablation study + controlled public-leaderboard variant generation.

Ablations are run *on top of one candidate-generation pass* (candidates and the
full feature matrix are computed once), so each experiment is cheap. Feature
groups are selected by column masks derived from ``FEATURE_NAMES``.
"""
from __future__ import annotations

import csv
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from .calibration import fit_calibrators
from .candidates import CandidatePair, evaluate_candidate_recall, generate_candidates
from .decision import (
    DecisionConfig,
    apply_decision,
    optimize_ambiguity,
    optimize_source_thresholds,
    optimize_threshold,
)
from .evaluation import evaluate
from .features import FEATURE_INDEX, FEATURE_NAMES, FeatureExtractor, labels_for_pairs
from .hard_negative_mining import MiningConfig, mine_hard_negatives
from .consistency import ConsistencyConfig, apply_consistency
from .training import TrainConfig, train_lightgbm
from .utils import LOG, PeakMemory, human_seconds


NAME_COLS = [
    "name_exact", "name_heavy_exact", "name_core_exact", "name_ratio", "name_jaro_winkler",
    "name_partial_ratio", "name_token_sort_ratio", "name_token_set_ratio",
    "name_partial_token_set_ratio", "name_core_token_set_ratio", "name_prefix_sim",
    "name_suffix_sim", "name_len_diff", "name_len_ratio", "name_tok_count_diff",
    "name_common_tokens", "name_token_coverage", "name_core_token_coverage", "name_ngram_jaccard",
]
ADDR_COLS = [
    "addr_exact", "addr_heavy_exact", "addr_ratio", "addr_token_sort_ratio",
    "addr_token_set_ratio", "addr_token_coverage", "addr_len_diff", "addr_len_ratio",
]
NUM_COLS = [
    "addr_num_overlap", "house_number_match", "postal_match", "addr_digit_overlap",
    "addr_first_token_match", "addr_last_token_match", "addr_completeness_agree",
    "addr_missing_s1", "addr_missing_cand",
]
CROSS_COLS = [
    "name_x_addr", "name_addr_min", "name_addr_max", "name_high_addr_high",
    "name_high_number_match", "name_high_postal_match", "name_medium_addr_high",
    "name_present_both", "addr_present_both", "country_equal", "country_conflict",
    "country_present_both",
]
RARITY_COLS = [
    "name_freq_log_s1", "name_freq_log_cand", "name_is_rare", "core_name_freq_log_cand",
    "addr_freq_log_cand", "rare_name_token_count_s1", "rare_addr_token_count_s1",
    "idf_token_overlap", "token_overlap",
]
BLOCK_COLS = [n for n in FEATURE_NAMES if n.startswith("retrieved_by_")] + [
    "number_of_blocks", "name_retrieval_rank", "ngram_retrieval_rank",
]


def cols_to_mask(cols: Sequence[str]) -> np.ndarray:
    return np.array([FEATURE_INDEX[c] for c in cols], dtype=np.int64)


def _eval_cfg(
    pairs, probs, n_s1, cs, dec_cfg, val_entities, labels, s1
) -> Tuple[float, dict]:
    rows = [r for r, p in enumerate(pairs) if p.s1_index in val_entities]
    pv = probs[rows]
    val_pairs = [pairs[r] for r in rows]
    val_gt = {i: labels.positives.get(i, set()) for i in val_entities}
    preds = apply_decision(val_pairs, pv, n_s1, cs.order, cs.source_of, dec_cfg)
    res = evaluate(preds, val_gt, n_s1, s1_country=s1.country_norm,
                   cs_order=cs.order, cs_source_of=cs.source_of)
    return res.macro_f05, res


def run_ablations(pipeline, cfg, out_csv: Optional[Path] = None) -> List[dict]:
    """Execute the 12-experiment ablation ladder and write the results table."""
    from . import indexing as idx
    from .blocking import build_blocking_index

    LOG.info("=== ABLATION STUDY ===")
    t0 = time.perf_counter()
    s1, cand_tables, cs, stats, labels = pipeline._prepare_train()
    block_cfg = pipeline.section_blocking() if hasattr(pipeline, "section_blocking") else None
    if block_cfg is None:
        from .pipeline import blocking_config_from

        block_cfg = blocking_config_from(cfg.section("blocking"))
    bi = build_blocking_index(cs, stats, block_cfg)
    pairs = generate_candidates(s1, bi, block_cfg)
    extractor = FeatureExtractor(s1, cs, stats, bi, rare_df=block_cfg.rare_name_token_df)
    X = extractor.extract(pairs)
    y = labels_for_pairs(pairs, labels.positives)

    # split
    frac = float(cfg.section("split").get("train_fraction", 0.8))
    rng = np.random.default_rng(cfg.seed)
    order = np.arange(s1.n); rng.shuffle(order)
    n_train = int(round(frac * s1.n))
    train_e = set(order[:n_train].tolist()); val_e = set(order[n_train:].tolist())
    pair_s1 = np.array([p.s1_index for p in pairs])
    tr_rows = np.nonzero(np.isin(pair_s1, list(train_e)))[0]

    results: List[dict] = []
    train_cfg = TrainConfig(**{k: v for k, v in cfg.section("training").items()
                               if k in TrainConfig.__dataclass_fields__})

    def record(name: str, f05: float, details: str = "") -> None:
        results.append({"experiment": name, "macro_f05": round(f05, 6), "details": details})
        LOG.info("  [ablation] %-42s macro F0.5=%.5f  %s", name, f05, details)

    def train_and_eval(mask: Optional[np.ndarray], mining: bool = True,
                       calibrated: bool = False, optimize: bool = True,
                       use_consistency: bool = False) -> Tuple[float, dict]:
        Xsub = X if mask is None else X[:, mask]
        # hard-negative mining judges difficulty from the *full* feature space, so
        # it always uses original column indices; only the selected rows are reused.
        if mining:
            sel, _ = mine_hard_negatives(X[tr_rows], y[tr_rows], MiningConfig(seed=cfg.seed))
            Xtr, ytr = Xsub[tr_rows][sel], y[tr_rows][sel]
        else:
            Xtr, ytr = Xsub[tr_rows], y[tr_rows]
        val_rows = np.nonzero(~np.isin(pair_s1, list(train_e)))[0]
        val_pairs = [pairs[r] for r in val_rows]
        Xva, yva = Xsub[val_rows], y[val_rows]
        booster = train_lightgbm(Xtr, ytr, Xva, yva, train_cfg, feature_names=(
            FEATURE_NAMES if mask is None else [FEATURE_NAMES[i] for i in mask]))
        probs = booster.predict(Xsub, num_iteration=getattr(booster, "best_iteration", None))
        if calibrated:
            cals = fit_calibrators(yva, probs[val_rows])
            # pick best calibrator at 0.5 on the validation fold
            best = ("none", -1.0)
            for k, c in cals.items():
                s, _ = _eval_cfg(pairs, c.transform(probs), s1.n, cs,
                                 DecisionConfig(threshold=0.5), val_e, labels, s1)
                if s > best[1]:
                    best = (k, s)
            probs = cals[best[0]].transform(probs)
        dec = DecisionConfig(threshold=0.5)
        if optimize:
            pv = probs[val_rows]
            tsr = optimize_threshold(val_pairs, pv, {i: labels.positives.get(i, set()) for i in val_e},
                                     s1.n, cs.order, cs.source_of, s1_country=s1.country_norm,
                                     fine=False)
            dec = tsr.best
        f05, res = _eval_cfg(pairs, probs, s1.n, cs, dec, val_e, labels, s1)
        if use_consistency:
            pv = probs[val_rows]
            preds = apply_decision(val_pairs, pv, s1.n, cs.order, cs.source_of, dec)
            preds2 = apply_consistency(preds, val_pairs, pv, s1.n, cs.order, cs.source_of,
                                       ConsistencyConfig(enabled=True))
            val_gt = {i: labels.positives.get(i, set()) for i in val_e}
            res2 = evaluate(preds2, val_gt, s1.n, s1_country=s1.country_norm,
                            cs_order=cs.order, cs_source_of=cs.source_of)
            f05 = res2.macro_f05
        return f05, res

    # 1) exact matching only (rule: normalized name exact) ---------------
    exact_preds: Dict[int, Set[int]] = {}
    for p in pairs:
        if p.blocks & 1:  # exact_name block
            exact_preds.setdefault(p.s1_index, set()).add(p.gid)
    val_gt = {i: labels.positives.get(i, set()) for i in val_e}
    res1 = evaluate(exact_preds, val_gt, s1.n, s1_country=s1.country_norm,
                    cs_order=cs.order, cs_source_of=cs.source_of)
    record("1. exact matching only", res1.macro_f05, "rule-based, no model")

    # 2) exact + normalized-name/core match ------------------------------
    core_preds: Dict[int, Set[int]] = {}
    for p in pairs:
        if p.blocks & (1 | 2):
            core_preds.setdefault(p.s1_index, set()).add(p.gid)
    res2 = evaluate(core_preds, val_gt, s1.n, s1_country=s1.country_norm,
                    cs_order=cs.order, cs_source_of=cs.source_of)
    record("2. + normalization (core name)", res2.macro_f05, "rule-based")

    # 3) + fuzzy name ----------------------------------------------------
    f3, _ = train_and_eval(cols_to_mask(NAME_COLS), optimize=False)
    record("3. + fuzzy name features", f3)

    # 4) + fuzzy address -------------------------------------------------
    f4, _ = train_and_eval(cols_to_mask(NAME_COLS + ADDR_COLS), optimize=False)
    record("4. + fuzzy address features", f4)

    # 5) + numeric/address features --------------------------------------
    f5, _ = train_and_eval(cols_to_mask(NAME_COLS + ADDR_COLS + NUM_COLS), optimize=False)
    record("5. + numeric/address features", f5)

    # 6) + multi-blocking provenance -------------------------------------
    f6, _ = train_and_eval(cols_to_mask(NAME_COLS + ADDR_COLS + NUM_COLS + BLOCK_COLS), optimize=False)
    record("6. + multi-blocking features", f6)

    # 7) + rarity --------------------------------------------------------
    f7, _ = train_and_eval(cols_to_mask(NAME_COLS + ADDR_COLS + NUM_COLS + BLOCK_COLS + RARITY_COLS),
                           optimize=False)
    record("7. + rarity features", f7)

    # 8) + hard negatives ------------------------------------------------
    f8, _ = train_and_eval(None, mining=True, optimize=False)
    record("8. all features + hard negatives", f8)

    # 9) + calibrated threshold ------------------------------------------
    f9, _ = train_and_eval(None, mining=True, calibrated=True, optimize=True)
    record("9. + calibration + threshold opt", f9)

    # 10) + singleton decision layer -------------------------------------
    val_rows = np.nonzero(~np.isin(pair_s1, list(train_e)))[0]
    Xtr, ytr = X[tr_rows], y[tr_rows]
    sel, _ = mine_hard_negatives(Xtr, ytr, MiningConfig(seed=cfg.seed))
    booster = train_lightgbm(Xtr[sel], ytr[sel], X[val_rows], y[val_rows], train_cfg)
    probs = booster.predict(X, num_iteration=getattr(booster, "best_iteration", None))
    cals = fit_calibrators(y[val_rows], probs[val_rows])
    best_k, best_s = "none", -1
    for k, c in cals.items():
        s_, _ = _eval_cfg(pairs, c.transform(probs), s1.n, cs, DecisionConfig(threshold=0.5), val_e, labels, s1)
        if s_ > best_s:
            best_k, best_s = k, s_
    probs_cal = cals[best_k].transform(probs)
    val_pairs = [pairs[r] for r in val_rows]
    pv = probs_cal[val_rows]
    tsr = optimize_threshold(val_pairs, pv, val_gt, s1.n, cs.order, cs.source_of,
                             s1_country=s1.country_norm, fine=True)
    dec10 = tsr.best
    f10, _ = _eval_cfg(pairs, probs_cal, s1.n, cs, dec10, val_e, labels, s1)
    record("10. + tuned global threshold", f10, f"thr={dec10.threshold:.4f}")

    src_dec, f11, _ = optimize_source_thresholds(val_pairs, pv, val_gt, s1.n, cs.order,
                                                 cs.source_of, dec10, s1_country=s1.country_norm)
    record("11. + source-specific thresholds", f11,
           f"s2={src_dec.threshold_s2} s3={src_dec.threshold_s3}")

    amb_dec, f12, _ = optimize_ambiguity(val_pairs, pv, val_gt, s1.n, cs.order,
                                         cs.source_of, src_dec, s1_country=s1.country_norm)
    record("12. + ambiguity logic", f12,
           f"margin={amb_dec.margin} top_only={amb_dec.top_only} max={amb_dec.max_matches_per_entity}")

    preds = apply_decision(val_pairs, pv, s1.n, cs.order, cs.source_of, amb_dec)
    preds_c = apply_consistency(preds, val_pairs, pv, s1.n, cs.order, cs.source_of,
                                ConsistencyConfig(enabled=True))
    res_c = evaluate(preds_c, val_gt, s1.n, s1_country=s1.country_norm,
                     cs_order=cs.order, cs_source_of=cs.source_of)
    record("13. + S2/S3 consistency (optional)", res_c.macro_f05,
           "only kept if > experiment 12")

    out_csv = out_csv or (cfg.reports_dir / "ablation_results.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["experiment", "macro_f05", "details"])
        w.writeheader()
        for r in results:
            w.writerow(r)
    LOG.info("ablation results -> %s (%.1fs)", out_csv, time.perf_counter() - t0)
    return results


# --------------------------------------------------------------------------- #
# Public leaderboard strategy variants
# --------------------------------------------------------------------------- #
def build_variants(base_decision: DecisionConfig, probs: np.ndarray, pairs, s1, cs, labels,
                   val_e, val_gt) -> List[dict]:
    """Five controlled submission variants (A..E), each a documented decision point."""
    variants = []

    def add(name, dec: DecisionConfig, note: str):
        f05, res = _eval_cfg(pairs, probs, s1.n, cs, dec, val_e, labels, s1)
        variants.append({"variant": name, "macro_f05": round(f05, 6),
                         "decision": dec.__dict__, "note": note})

    add("A conservative", DecisionConfig(**{**base_decision.__dict__, "threshold": min(0.95, base_decision.threshold + 0.1)}),
        "higher threshold -> fewer false merges")
    add("B balanced", base_decision, "validated optimum")
    thr_lo = max(0.3, base_decision.threshold - 0.05)
    add("C adaptive (lower thr)", DecisionConfig(**{**base_decision.__dict__, "threshold": thr_lo}),
        "more recall, more false merges")
    add("D best-validation", base_decision, "identical to B, recorded for the 5-submission budget")
    add("E top-only", DecisionConfig(**{**base_decision.__dict__, "top_only": True}),
        "at most one match per entity")
    return variants
