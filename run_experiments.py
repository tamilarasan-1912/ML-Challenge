#!/usr/bin/env python3
"""Experiment runner: ablations, variant comparison, tuning.

Examples
--------
    python run_experiments.py ablation --data-root student_resource
    python run_experiments.py variants --data-root student_resource
    python run_experiments.py tune     --data-root student_resource --trials 25
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.pipeline import Pipeline, load_config, train_config_from  # noqa: E402
from src.utils import LOG, human_seconds  # noqa: E402


def cmd_ablation(pipeline: Pipeline) -> None:
    from src.ablation import run_ablations

    run_ablations(pipeline, pipeline.cfg)


def cmd_variants(pipeline: Pipeline) -> None:
    import numpy as np

    from src import indexing as idx
    from src.ablation import build_variants
    from src.blocking import BlockingConfig, build_blocking_index
    from src.calibration import Calibrator
    from src.candidates import generate_candidates
    from src.decision import DecisionConfig
    from src.features import FeatureExtractor

    import lightgbm as lgb
    import pickle

    bundle = json.loads((pipeline.cfg.models_dir / "artifacts_bundle.json").read_text())
    dec = DecisionConfig(**bundle["decision"])
    block_cfg = BlockingConfig(**bundle["blocking"])
    s1, cand_tables, cs, stats, labels = pipeline._prepare_train()
    booster = lgb.Booster(model_file=str(pipeline.cfg.models_dir / "lgbm_pair_model.txt"))
    with open(pipeline.cfg.models_dir / "calibrator.pkl", "rb") as fh:
        cal: Calibrator = pickle.load(fh)
    bi = build_blocking_index(cs, stats, block_cfg)
    pairs = generate_candidates(s1, bi, block_cfg)
    X = FeatureExtractor(s1, cs, stats, bi, rare_df=block_cfg.rare_name_token_df).extract(pairs)
    probs = cal.transform(booster.predict(X, num_iteration=bundle.get("best_iteration") or None))

    rng = np.random.default_rng(int(bundle.get("seed", 42)))
    order = np.arange(s1.n); rng.shuffle(order)
    n_train = int(round(float(bundle.get("train_fraction", 0.8)) * s1.n))
    val_e = set(order[n_train:].tolist())
    val_gt = {i: labels.positives.get(i, set()) for i in val_e}
    variants = build_variants(dec, probs, pairs, s1, cs, labels, val_e, val_gt)
    out = pipeline.cfg.reports_dir / "leaderboard_variants.csv"
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["variant", "macro_f05", "threshold", "threshold_s2", "threshold_s3",
                    "margin", "top_only", "max_matches", "note"])
        for v in variants:
            d = v["decision"]
            w.writerow([v["variant"], v["macro_f05"], d["threshold"], d["threshold_s2"],
                        d["threshold_s3"], d["margin"], d["top_only"],
                        d["max_matches_per_entity"], v["note"]])
    LOG.info("wrote variants to %s", out)
    for v in variants:
        LOG.info("  %-20s macro F0.5=%.5f", v["variant"], v["macro_f05"])


def cmd_tune(pipeline: Pipeline, trials: int, budget: float) -> None:
    import numpy as np

    from src import indexing as idx
    from src.blocking import blocking_config_from, build_blocking_index
    from src.candidates import generate_candidates
    from src.decision import DecisionConfig, optimize_threshold
    from src.evaluation import evaluate
    from src.features import FeatureExtractor, labels_for_pairs
    from src.hard_negative_mining import MiningConfig, mine_hard_negatives
    from src.training import TrainConfig, train_lightgbm
    from src.decision import apply_decision

    s1, cand_tables, cs, stats, labels = pipeline._prepare_train()
    block_cfg = blocking_config_from(pipeline.cfg.section("blocking"))
    bi = build_blocking_index(cs, stats, block_cfg)
    pairs = generate_candidates(s1, bi, block_cfg)
    X = FeatureExtractor(s1, cs, stats, bi, rare_df=block_cfg.rare_name_token_df).extract(pairs)
    y = labels_for_pairs(pairs, labels.positives)
    pair_s1 = np.array([p.s1_index for p in pairs])
    frac = float(pipeline.cfg.section("split").get("train_fraction", 0.8))
    rng = np.random.default_rng(pipeline.cfg.seed)
    order = np.arange(s1.n); rng.shuffle(order)
    n_train = int(round(frac * s1.n))
    train_e = set(order[:n_train].tolist()); val_e = set(order[n_train:].tolist())
    tr = np.nonzero(np.isin(pair_s1, list(train_e)))[0]
    va = np.nonzero(np.isin(pair_s1, list(val_e)))[0]
    sel, _ = mine_hard_negatives(X[tr], y[tr], MiningConfig(seed=pipeline.cfg.seed))
    Xtr, ytr = X[tr][sel], y[tr][sel]
    val_gt = {i: labels.positives.get(i, set()) for i in val_e}

    spaces = {
        "num_leaves": [31, 63, 96, 160, 255],
        "learning_rate": [0.02, 0.03, 0.05, 0.08, 0.12],
        "min_child_samples": [10, 20, 40, 80],
        "colsample_bytree": [0.6, 0.75, 0.9, 1.0],
        "reg_lambda": [0.0, 1.0, 5.0, 20.0],
        "subsample": [0.7, 0.85, 1.0],
    }
    base = train_config_from(pipeline.cfg.section("training"))
    rs = np.random.default_rng(pipeline.cfg.seed + 7)
    results = []
    t0 = time.perf_counter()
    for t in range(trials):
        if time.perf_counter() - t0 > budget:
            LOG.info("tuning budget exhausted after %d trials", t)
            break
        params = {k: v[int(rs.integers(len(v)))] for k, v in spaces.items()}
        tc = TrainConfig(**{**base.__dict__, **params})
        booster = train_lightgbm(Xtr, ytr, X[va], y[va], tc)
        probs = booster.predict(X, num_iteration=getattr(booster, "best_iteration", None))
        tsr = optimize_threshold(pairs, probs, val_gt, s1.n, cs.order, cs.source_of,
                                 s1_country=s1.country_norm, fine=False)
        preds = apply_decision([pairs[r] for r in va], probs[va], s1.n, cs.order, cs.source_of, tsr.best)
        res = evaluate(preds, val_gt, s1.n, s1_country=s1.country_norm,
                       cs_order=cs.order, cs_source_of=cs.source_of)
        results.append({**params, "macro_f05": res.macro_f05, "threshold": tsr.best.threshold})
        LOG.info("trial %d/%d: %s -> F0.5=%.5f", t + 1, trials, params, res.macro_f05)

    results.sort(key=lambda r: -r["macro_f05"])
    out = pipeline.cfg.experiments_dir / "tuning_results.csv"
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(results[0].keys()))
        w.writeheader()
        for r in results:
            w.writerow(r)
    LOG.info("best tuning result: %s", results[0])
    LOG.info("wrote %s", out)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Amazon ML Challenge 2026 - experiments")
    p.add_argument("command", choices=["ablation", "variants", "tune"])
    p.add_argument("--data-root", default=None)
    p.add_argument("--config", default=str(Path(__file__).resolve().parent / "config.yaml"))
    p.add_argument("--trials", type=int, default=25)
    p.add_argument("--budget", type=float, default=1800.0)
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    pipeline = Pipeline(cfg, data_root=args.data_root)
    t0 = time.perf_counter()
    if args.command == "ablation":
        cmd_ablation(pipeline)
    elif args.command == "variants":
        cmd_variants(pipeline)
    elif args.command == "tune":
        cmd_tune(pipeline, args.trials, args.budget)
    LOG.info("done in %s", human_seconds(time.perf_counter() - t0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
