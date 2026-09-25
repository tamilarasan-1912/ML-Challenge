"""Pipeline orchestration: ties every stage together.

Stages
------
profile             : measure the real dataset -> reports/data_profile.*
train               : labels -> candidates -> features -> hard negatives ->
                      LightGBM -> calibration -> threshold/ambiguity search
validate            : evaluate the trained model on the entity-level validation
                      fold and emit validation_results.csv / error_analysis.csv
predict             : run test inference and write the two submission TSVs
validate-submission : run the official validator + internal structural checks
all                 : profile -> train -> validate -> predict -> validate-submission
"""
from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import polars as pl

from . import io_utils as io
from . import indexing as idx
from .blocking import BlockingConfig, build_blocking_index
from .calibration import Calibrator, fit_calibrators
from .candidates import (
    CandidatePair,
    evaluate_candidate_recall,
    generate_candidates,
)
from .decision import (
    DecisionConfig,
    apply_decision,
    group_pairs_by_entity,
    optimize_ambiguity,
    optimize_source_thresholds,
    optimize_threshold,
)
from .evaluation import EvalResult, evaluate
from .features import FEATURE_NAMES, FeatureExtractor, labels_for_pairs
from .hard_negative_mining import MiningConfig, mine_hard_negatives
from .indexing import CandidateSpace, FrequencyStats, SourceTable, build_source_table
from .io_utils import parse_match_list
from .labels import LabelSet, build_labels
from .submission import (
    internal_checks,
    write_candidate_pairs,
    write_matching_results,
)
from .training import TrainConfig, feature_importance_table, train_lightgbm
from .utils import (
    LOG,
    PeakMemory,
    ensure_dir,
    human_seconds,
    read_json,
    set_seed,
    timed,
    write_json,
)


# --------------------------------------------------------------------------- #
# Config helpers
# --------------------------------------------------------------------------- #
class Config:
    def __init__(self, raw: dict) -> None:
        self.raw = raw
        self.seed = int(raw.get("seed", 42))
        p = raw.get("paths", {})
        self.output_dir = Path(p.get("output_dir", "output"))
        self.reports_dir = Path(p.get("reports_dir", "reports"))
        self.models_dir = Path(p.get("models_dir", "models"))
        self.experiments_dir = Path(p.get("experiments_dir", "experiments"))
        self.artifacts_dir = Path(p.get("artifacts_dir", "artifacts"))

    def section(self, name: str) -> dict:
        return dict(self.raw.get(name, {}) or {})


def load_config(path: Optional[str]) -> Config:
    import yaml

    raw: dict = {}
    if path and Path(path).exists():
        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
    return Config(raw)


def blocking_config_from(cfg_dict: dict) -> BlockingConfig:
    allowed = set(BlockingConfig.__dataclass_fields__.keys())
    return BlockingConfig(**{k: v for k, v in cfg_dict.items() if k in allowed})


def train_config_from(cfg_dict: dict) -> TrainConfig:
    allowed = set(TrainConfig.__dataclass_fields__.keys())
    return TrainConfig(**{k: v for k, v in cfg_dict.items() if k in allowed})


def mining_config_from(cfg_dict: dict, seed: int) -> MiningConfig:
    allowed = set(MiningConfig.__dataclass_fields__.keys())
    d = {k: v for k, v in cfg_dict.items() if k in allowed}
    d["seed"] = seed
    return MiningConfig(**d)


def decision_config_from(cfg_dict: dict) -> DecisionConfig:
    allowed = set(DecisionConfig.__dataclass_fields__.keys())
    return DecisionConfig(**{k: v for k, v in cfg_dict.items() if k in allowed})


# --------------------------------------------------------------------------- #
# Data assembly
# --------------------------------------------------------------------------- #
def _load_tables(paths_map: Dict[str, Path], max_rows: Optional[int] = None) -> Dict[str, SourceTable]:
    tables: Dict[str, SourceTable] = {}
    for tag, p in paths_map.items():
        t0 = time.perf_counter()
        df = io.load_source(p, n_rows=max_rows)
        st = build_source_table(tag, df)
        tables[tag] = st
        LOG.info("loaded+represented %s: %d rows in %.2fs", tag, st.n, time.perf_counter() - t0)
        del df
    return tables


def _frames_for_gt_reference(paths_map: Dict[str, Path], keep_in_memory: bool = False):
    """Nothing extra needed; tables already carry raw + normalized fields."""
    return None


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
class Pipeline:
    def __init__(self, cfg: Config, data_root: Optional[str] = None) -> None:
        self.cfg = cfg
        self.data_root = data_root or cfg.raw.get("paths", {}).get("data_root")
        self.paths = io.resolve_dataset(self.data_root)
        LOG.info("resolved dataset:\n%s", self.paths.describe())
        set_seed(cfg.seed)
        ensure_dir(cfg.output_dir)
        ensure_dir(cfg.reports_dir)
        ensure_dir(cfg.models_dir)
        ensure_dir(cfg.experiments_dir)
        ensure_dir(cfg.artifacts_dir)

    # ---------------- profile ---------------- #
    def profile(self) -> dict:
        from .data_profile import build_profile

        prof = build_profile(self.data_root, str(self.cfg.reports_dir))
        return prof

    # ---------------- shared: build train candidates ---------------- #
    def _prepare_train(self):
        tables = _load_tables(self.paths.train)
        if "S1" not in tables:
            raise RuntimeError("train_source1.tsv missing")
        s1 = tables["S1"]
        cand_tables = {k: v for k, v in tables.items() if k in ("S2", "S3")}
        cs = idx.build_candidate_space(cand_tables, order=("S2", "S3"))
        stats = idx.build_frequency_stats([s1] + list(cand_tables.values()))
        gt = io.load_ground_truth(self.paths.train_gt)
        labels = build_labels(gt, s1, cs)
        return s1, cand_tables, cs, stats, labels

    # ---------------- train ---------------- #
    def train(self) -> dict:
        t_start = time.perf_counter()
        mem = PeakMemory(); mem.__enter__()
        timings: Dict[str, float] = {}

        with timed("prepare_train", timings):
            s1, cand_tables, cs, stats, labels = self._prepare_train()

        block_cfg = blocking_config_from(self.cfg.section("blocking"))
        with timed("build_indexes", timings):
            bi = build_blocking_index(cs, stats, block_cfg)
        with timed("generate_candidates", timings):
            pairs = generate_candidates(s1, bi, block_cfg)
        LOG.info("candidate pairs: %d (%.2f/S1)", len(pairs), len(pairs) / max(1, s1.n))

        recall_rep, counts = evaluate_candidate_recall(s1, cs, pairs, labels.positives)
        LOG.info(
            "candidate recall=%.4f reduction=%.6f mean/S1=%.2f p99=%.0f",
            recall_rep.candidate_recall, recall_rep.reduction_ratio,
            recall_rep.mean_candidates, recall_rep.p99_candidates,
        )
        write_json(self.cfg.reports_dir / "candidate_recall.json", recall_rep.as_dict())

        with timed("extract_features", timings):
            extractor = FeatureExtractor(s1, cs, stats, bi, rare_df=block_cfg.rare_name_token_df)
            X = extractor.extract(pairs)
        y = labels_for_pairs(pairs, labels.positives)

        # ---------------- entity-level split ---------------- #
        split_cfg = self.cfg.section("split")
        frac = float(split_cfg.get("train_fraction", 0.8))
        rng = np.random.default_rng(self.cfg.seed)
        all_idx = np.arange(s1.n)
        rng.shuffle(all_idx)
        n_train = int(round(frac * s1.n))
        train_entities = set(all_idx[:n_train].tolist())
        val_entities = set(all_idx[n_train:].tolist())

        pair_s1 = np.array([p.s1_index for p in pairs], dtype=np.int64)
        is_train_row = np.array([s in train_entities for s in pair_s1], dtype=bool)
        train_rows = np.nonzero(is_train_row)[0]
        val_rows = np.nonzero(~is_train_row)[0]
        LOG.info("split: %d train entities (%d pairs), %d val entities (%d pairs)",
                 len(train_entities), len(train_rows), len(val_entities), len(val_rows))

        y_train_raw = y[train_rows]
        X_train_raw = X[train_rows]
        mining_cfg = mining_config_from(self.cfg.section("hard_negative_mining"), self.cfg.seed)
        sel, mining_info = mine_hard_negatives(X_train_raw, y_train_raw, mining_cfg)
        X_train = X_train_raw[sel]
        y_train = y_train_raw[sel]

        X_val = X[val_rows]
        y_val = y[val_rows]

        train_cfg = train_config_from(self.cfg.section("training"))
        with timed("train_lightgbm", timings):
            booster = train_lightgbm(
                X_train, y_train, X_val, y_val, train_cfg, model_dir=str(self.cfg.models_dir)
            )
        LOG.info("best iteration: %s", getattr(booster, "best_iteration", None))

        imp = feature_importance_table(booster, top_k=40)
        with open(self.cfg.reports_dir / "feature_importance.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh); w.writerow(["feature", "gain", "split"])
            for name, g, s in imp:
                w.writerow([name, f"{g:.3f}", int(s)])

        # ---------------- calibrate on validation pairs ---------------- #
        probs_val_raw = booster.predict(X_val, num_iteration=getattr(booster, "best_iteration", None))
        opt = self.cfg.section("optimization")
        calibrator = Calibrator("none")
        calib_table: List[dict] = []
        if opt.get("calibrate", True):
            cals = fit_calibrators(y_val, probs_val_raw)
            best_kind, best_sc = "none", -1.0
            val_pairs = [pairs[r] for r in val_rows]
            val_gt = {i: labels.positives.get(i, set()) for i in val_entities}
            for kind, cal in cals.items():
                pv = cal.transform(probs_val_raw)
                cfg0 = DecisionConfig(threshold=0.5)
                preds = apply_decision(val_pairs, pv, s1.n, cs.order, cs.source_of, cfg0)
                res = evaluate(preds, val_gt, s1.n, s1_country=s1.country_norm,
                               cs_order=cs.order, cs_source_of=cs.source_of)
                calib_table.append({"kind": kind, "macro_f05_at_0.5": res.macro_f05})
                if res.macro_f05 > best_sc:
                    best_sc, best_kind = res.macro_f05, kind
            calibrator = cals[best_kind]
            LOG.info("calibration selected: %s (macro F0.5@0.5 = %.5f)", best_kind, best_sc)

        probs_val = calibrator.transform(probs_val_raw)
        val_pairs = [pairs[r] for r in val_rows]
        val_gt = {i: labels.positives.get(i, set()) for i in val_entities}

        # ---------------- threshold + decision optimisation ---------------- #
        with timed("optimize_threshold", timings):
            tsr = optimize_threshold(
                val_pairs, probs_val, val_gt, s1.n, cs.order, cs.source_of,
                s1_country=s1.country_norm, fine=bool(opt.get("fine_threshold", True)),
            )
        dec_cfg = tsr.best
        src_table: List[dict] = []
        if opt.get("optimize_source_thresholds", True):
            dec_cfg, sc, src_table = optimize_source_thresholds(
                val_pairs, probs_val, val_gt, s1.n, cs.order, cs.source_of,
                dec_cfg, s1_country=s1.country_norm,
            )
        amb_table: List[dict] = []
        if opt.get("optimize_ambiguity", True):
            dec_cfg, sc, amb_table = optimize_ambiguity(
                val_pairs, probs_val, val_gt, s1.n, cs.order, cs.source_of,
                dec_cfg, s1_country=s1.country_norm,
            )

        final_preds = apply_decision(val_pairs, probs_val, s1.n, cs.order, cs.source_of, dec_cfg)
        final_eval = evaluate(final_preds, val_gt, s1.n, s1_country=s1.country_norm,
                              cs_order=cs.order, cs_source_of=cs.source_of)
        LOG.info(
            "VALIDATION macro F0.5=%.5f P=%.5f R=%.5f singleton=%.5f",
            final_eval.macro_f05, final_eval.precision, final_eval.recall, final_eval.singleton_score,
        )

        # ---------------- persist artifacts ---------------- #
        bundle = {
            "feature_names": FEATURE_NAMES,
            "decision": dec_cfg.__dict__,
            "blocking": asdict(block_cfg),
            "mining": asdict(mining_cfg),
            "calibrator": calibrator.as_dict(),
            "best_iteration": int(getattr(booster, "best_iteration", 0) or 0),
            "train_fraction": frac,
            "seed": self.cfg.seed,
            "metrics": final_eval.as_dict().copy(),
            "candidate_recall": recall_rep.as_dict(),
        }
        # calibrator needs to be persisted for reuse
        import pickle

        with open(self.cfg.models_dir / "artifacts_bundle.json", "w", encoding="utf-8") as fh:
            json.dump(bundle, fh, indent=2, default=str)
        with open(self.cfg.models_dir / "calibrator.pkl", "wb") as fh:
            pickle.dump(calibrator, fh)

        # threshold / calibration tables
        with open(self.cfg.reports_dir / "threshold_search.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["threshold", "macro_f05"], extrasaction="ignore")
            w.writeheader()
            for r in tsr.table:
                w.writerow(r)
        with open(self.cfg.reports_dir / "calibration_search.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["kind", "macro_f05_at_0.5"], extrasaction="ignore")
            w.writeheader()
            for r in calib_table:
                w.writerow(r)

        # validation results
        self._write_validation_results(final_eval, dec_cfg)
        self._write_error_analysis(val_pairs, probs_val, val_gt, s1, cs, dec_cfg)

        runtime = time.perf_counter() - t_start
        mem.__exit__()
        timings["total"] = runtime
        self._log_experiment(
            experiment_id=f"train_{int(time.time())}",
            recall_rep=recall_rep,
            eval_res=final_eval,
            dec_cfg=dec_cfg,
            train_size=len(X_train),
            val_size=len(X_val),
            runtime=runtime,
            peak_mem=mem.peak,
            notes=f"calib={calibrator.kind};mining_sel={mining_info['n_selected']}",
        )
        LOG.info("train complete in %s; %s", human_seconds(runtime), mem.report())
        return {
            "eval": final_eval,
            "recall": recall_rep,
            "decision": dec_cfg,
            "runtime": runtime,
            "peak_mem": mem.peak,
            "timings": timings,
            "mining": mining_info,
        }

    # ---------------- validate ---------------- #
    def validate(self) -> dict:
        """Evaluate the saved model + decision config on the held-out fold."""
        return self._run_validation_only(emit_files=True)

    def _run_validation_only(self, emit_files: bool = True) -> dict:
        import pickle

        s1, cand_tables, cs, stats, labels = self._prepare_train()
        bundle = read_json(self.cfg.models_dir / "artifacts_bundle.json")
        dec_cfg = DecisionConfig(**bundle["decision"])
        block_cfg = BlockingConfig(**bundle["blocking"])
        frac = float(bundle.get("train_fraction", 0.8))

        import lightgbm as lgb

        booster = lgb.Booster(model_file=str(self.cfg.models_dir / "lgbm_pair_model.txt"))
        with open(self.cfg.models_dir / "calibrator.pkl", "rb") as fh:
            calibrator: Calibrator = pickle.load(fh)

        rng = np.random.default_rng(int(bundle.get("seed", 42)))
        all_idx = np.arange(s1.n)
        rng.shuffle(all_idx)
        n_train = int(round(frac * s1.n))
        val_entities = set(all_idx[n_train:].tolist())

        bi = build_blocking_index(cs, stats, block_cfg)
        pairs = generate_candidates(s1, bi, block_cfg)
        extractor = FeatureExtractor(s1, cs, stats, bi, rare_df=block_cfg.rare_name_token_df)
        X = extractor.extract(pairs)
        probs = calibrator.transform(booster.predict(X, num_iteration=bundle.get("best_iteration") or None))

        val_pairs = [p for p in pairs if p.s1_index in val_entities]
        rows = [r for r, p in enumerate(pairs) if p.s1_index in val_entities]
        pv = probs[rows]
        val_gt = {i: labels.positives.get(i, set()) for i in val_entities}
        preds = apply_decision(val_pairs, pv, s1.n, cs.order, cs.source_of, dec_cfg)
        res = evaluate(preds, val_gt, s1.n, s1_country=s1.country_norm,
                       cs_order=cs.order, cs_source_of=cs.source_of)
        LOG.info("validate: macro F0.5=%.5f P=%.5f R=%.5f", res.macro_f05, res.precision, res.recall)
        if emit_files:
            self._write_validation_results(res, dec_cfg)
            self._write_error_analysis(val_pairs, pv, val_gt, s1, cs, dec_cfg)
        return {"eval": res, "decision": dec_cfg}

    # ---------------- predict ---------------- #
    def predict(self) -> dict:
        import pickle

        t0 = time.perf_counter()
        mem = PeakMemory(); mem.__enter__()

        bundle = read_json(self.cfg.models_dir / "artifacts_bundle.json")
        dec_cfg = DecisionConfig(**bundle["decision"])
        block_cfg = BlockingConfig(**bundle["blocking"])

        import lightgbm as lgb

        booster = lgb.Booster(model_file=str(self.cfg.models_dir / "lgbm_pair_model.txt"))
        with open(self.cfg.models_dir / "calibrator.pkl", "rb") as fh:
            calibrator: Calibrator = pickle.load(fh)

        LOG.info("loading TEST sources")
        tables = _load_tables(self.paths.test)
        s1 = tables["S1"]
        cand_tables = {k: v for k, v in tables.items() if k in ("S2", "S3")}
        cs = idx.build_candidate_space(cand_tables, order=("S2", "S3"))
        # frequency stats are recomputed from the *provided test data* only
        stats = idx.build_frequency_stats([s1] + list(cand_tables.values()))

        bi = build_blocking_index(cs, stats, block_cfg)
        pairs = generate_candidates(s1, bi, block_cfg)
        LOG.info("test candidate pairs: %d (%.2f/S1)", len(pairs), len(pairs) / max(1, s1.n))
        extractor = FeatureExtractor(s1, cs, stats, bi, rare_df=block_cfg.rare_name_token_df)
        X = extractor.extract(pairs)
        probs = calibrator.transform(booster.predict(X, num_iteration=bundle.get("best_iteration") or None))

        preds = apply_decision(pairs, probs, s1.n, cs.order, cs.source_of, dec_cfg)

        write_matching_results(self.cfg.output_dir / "matching_results.tsv", s1.entity_ids, preds, cs)
        write_candidate_pairs(self.cfg.output_dir / "candidate_pairs.tsv", s1.entity_ids, pairs, cs)

        n_matched = sum(len(v) for v in preds.values())
        n_nonempty = sum(1 for v in preds.values() if v)
        runtime = time.perf_counter() - t0
        mem.__exit__()
        LOG.info(
            "predict complete: %d S1 rows, %d with matches, %d accepted pairs, %s; %s",
            s1.n, n_nonempty, n_matched, human_seconds(runtime), mem.report(),
        )
        return {
            "n_s1": s1.n,
            "n_with_matches": n_nonempty,
            "n_accepted_pairs": n_matched,
            "runtime": runtime,
            "peak_mem": mem.peak,
        }

    # ---------------- official validator ---------------- #
    def validate_submission(self) -> dict:
        import subprocess
        import sys

        # 1) official validator
        official = {"ran": False, "returncode": None, "stdout": "", "stderr": "", "pass": None}
        if self.paths.validator is not None:
            cmd = [
                sys.executable,
                str(self.paths.validator),
                "--matching", str(self.cfg.output_dir / "matching_results.tsv"),
                "--candidate", str(self.cfg.output_dir / "candidate_pairs.tsv"),
                "--test-dir", str(self.paths.test["S1"].parent) if "S1" in self.paths.test else "dataset/test",
            ]
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
                official.update(
                    ran=True, returncode=proc.returncode,
                    stdout=proc.stdout[-8000:], stderr=proc.stderr[-8000:],
                    pass_=(proc.returncode == 0),
                )
            except Exception as exc:
                official.update(ran=True, returncode=-1, stderr=str(exc), pass_=False)
        else:
            LOG.warning("official validator not found next to the data; skipping")

        # 2) internal structural checks
        tables = _load_tables(self.paths.test)
        s1_ids = tables["S1"].entity_ids
        s2_ids = tables.get("S2", SourceTable("S2", [])).entity_ids
        s3_ids = tables.get("S3", SourceTable("S3", [])).entity_ids
        ok, problems = internal_checks(
            self.cfg.output_dir / "matching_results.tsv",
            self.cfg.output_dir / "candidate_pairs.tsv",
            s1_ids, s2_ids, s3_ids,
        )
        LOG.info("internal submission checks: %s (%d problems)", "PASS" if ok else "FAIL", len(problems))
        for p in problems[:30]:
            LOG.warning("  problem: %s", p)

        result = {"official": official, "internal_pass": ok, "internal_problems": problems}
        write_json(self.cfg.reports_dir / "submission_validation.json", result)

        # auto-fix path: if the official validator exposes a CLI mismatch, we
        # surface the exact stderr so the operator can adjust, rather than hiding it.
        return result

    # ---------------- reports ---------------- #
    def _write_validation_results(self, res: EvalResult, dec_cfg: DecisionConfig) -> None:
        path = self.cfg.reports_dir / "validation_results.csv"
        rows = [
            ("macro_f05", res.macro_f05),
            ("precision", res.precision),
            ("recall", res.recall),
            ("n_entities", res.n_entities),
            ("n_entities_with_gt", res.n_entities_with_gt),
            ("singleton_score", res.singleton_score),
            ("match_score", res.match_score),
            ("tp", res.tp), ("fp", res.fp), ("fn", res.fn),
            ("n_predicted_pairs", res.n_predicted_pairs),
            ("threshold", dec_cfg.threshold),
            ("threshold_s2", dec_cfg.threshold_s2 if dec_cfg.threshold_s2 is not None else ""),
            ("threshold_s3", dec_cfg.threshold_s3 if dec_cfg.threshold_s3 is not None else ""),
            ("high_conf", dec_cfg.high_conf),
            ("margin", dec_cfg.margin),
            ("top_only", dec_cfg.top_only),
            ("max_matches_per_entity", dec_cfg.max_matches_per_entity),
        ]
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh); w.writerow(["metric", "value"])
            for k, v in rows:
                w.writerow([k, v])
        write_json(self.cfg.reports_dir / "validation_results.json", {
            "eval": res.as_dict(), "decision": dec_cfg.__dict__,
        })

    def _write_error_analysis(
        self,
        pairs: Sequence[CandidatePair],
        probs: np.ndarray,
        gt: Dict[int, Set[int]],
        s1: SourceTable,
        cs: CandidateSpace,
        dec_cfg: DecisionConfig,
    ) -> None:
        preds = apply_decision(pairs, probs, s1.n, cs.order, cs.source_of, dec_cfg)
        groups = group_pairs_by_entity(pairs)
        path = self.cfg.reports_dir / "error_analysis.csv"
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow([
                "error_type", "s1_id", "candidate_id", "s1_name", "candidate_name",
                "s1_address", "candidate_address", "country", "probability", "true_label",
                "blocks", "n_blocks", "top_prob", "second_prob", "margin", "n_candidates",
            ])
            for i in sorted(groups.keys()):
                true_set = gt.get(i, set())
                pred_set = preds.get(i, set())
                rows = groups[i]
                scored = sorted(((float(probs[r]), pairs[r].gid, pairs[r]) for r in rows), key=lambda x: -x[0])
                top = scored[0][0] if scored else 0.0
                second = scored[1][0] if len(scored) > 1 else 0.0
                marg = top - second
                for prob, g, pr in scored:
                    is_true = g in true_set
                    is_pred = g in pred_set
                    if is_true and not is_pred:
                        etype = "false_negative"
                    elif is_pred and not is_true:
                        etype = "false_positive"
                    else:
                        continue
                    w.writerow([
                        etype, s1.entity_ids[i], cs.entity_ids[g],
                        s1.name_raw[i], cs.name_raw[g],
                        s1.addr_raw[i], cs.addr_raw[g],
                        s1.country_norm[i], f"{prob:.6f}", int(is_true),
                        pr.blocks, pr.n_blocks, f"{top:.6f}", f"{second:.6f}", f"{marg:.6f}", len(scored),
                    ])
        LOG.info("wrote error analysis to %s", path)

    def _log_experiment(self, **kw) -> None:
        path = self.cfg.experiments_dir / "experiment_log.csv"
        columns = [
            "experiment_id", "timestamp", "feature_version", "blocking_version", "model_version",
            "train_size", "validation_size", "candidate_recall", "candidate_reduction_ratio",
            "candidate_count_mean", "threshold", "S2_threshold", "S3_threshold",
            "macro_f05", "precision", "recall", "singleton_score",
            "false_positive_count", "false_negative_count", "runtime", "peak_memory", "notes",
        ]
        recall_rep = kw["recall_rep"]; res = kw["eval_res"]; dec = kw["dec_cfg"]
        row = {
            "experiment_id": kw["experiment_id"],
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "feature_version": f"v1_n{len(FEATURE_NAMES)}",
            "blocking_version": "multiblock_v1_10blocks",
            "model_version": "lightgbm_v1",
            "train_size": kw["train_size"],
            "validation_size": kw["val_size"],
            "candidate_recall": round(recall_rep.candidate_recall, 6),
            "candidate_reduction_ratio": round(recall_rep.reduction_ratio, 8),
            "candidate_count_mean": round(recall_rep.mean_candidates, 4),
            "threshold": dec.threshold,
            "S2_threshold": dec.threshold_s2 if dec.threshold_s2 is not None else "",
            "S3_threshold": dec.threshold_s3 if dec.threshold_s3 is not None else "",
            "macro_f05": round(res.macro_f05, 6),
            "precision": round(res.precision, 6),
            "recall": round(res.recall, 6),
            "singleton_score": round(res.singleton_score, 6),
            "false_positive_count": res.fp,
            "false_negative_count": res.fn,
            "runtime": round(kw["runtime"], 2),
            "peak_memory": int(kw["peak_mem"]),
            "notes": kw.get("notes", ""),
        }
        new_file = not path.exists()
        with open(path, "a", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=columns)
            if new_file:
                w.writeheader()
            w.writerow(row)
        LOG.info("appended experiment row to %s", path)

    # ---------------- all ---------------- #
    def run_all(self) -> dict:
        self.profile()
        train_res = self.train()
        val_res = self.validate()
        pred_res = self.predict()
        sub_res = self.validate_submission()
        from .reporting import generate_final_report

        report = generate_final_report(self.cfg.reports_dir, self.cfg.models_dir, self.cfg.output_dir)
        LOG.info("final report: %s", report)
        return {
            "train": train_res, "validate": val_res, "predict": pred_res,
            "submission": sub_res, "report": str(report),
        }
