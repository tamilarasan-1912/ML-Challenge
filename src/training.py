"""LightGBM training with grouped (entity-level) validation and tuning."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .features import FEATURE_NAMES, FEATURE_INDEX
from .utils import LOG, ensure_dir


@dataclass
class TrainConfig:
    num_leaves: int = 96
    max_depth: int = -1
    learning_rate: float = 0.05
    n_estimators: int = 3000
    min_child_samples: int = 20
    subsample: float = 0.9
    subsample_freq: int = 1
    colsample_bytree: float = 0.9
    reg_alpha: float = 0.1
    reg_lambda: float = 1.0
    min_split_gain: float = 0.0
    objective: str = "binary"
    seed: int = 42
    early_stopping_rounds: int = 150
    n_jobs: int = -1
    verbose_eval: int = 100


def train_lightgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cfg: TrainConfig,
    feature_names: Sequence[str] = tuple(FEATURE_NAMES),
    model_dir: Optional[str] = None,
):
    import lightgbm as lgb

    params = {
        "objective": cfg.objective,
        "metric": ["binary_logloss", "auc", "average_precision"],
        "num_leaves": cfg.num_leaves,
        "max_depth": cfg.max_depth,
        "learning_rate": cfg.learning_rate,
        "min_child_samples": cfg.min_child_samples,
        "subsample": cfg.subsample,
        "subsample_freq": cfg.subsample_freq,
        "colsample_bytree": cfg.colsample_bytree,
        "reg_alpha": cfg.reg_alpha,
        "reg_lambda": cfg.reg_lambda,
        "min_split_gain": cfg.min_split_gain,
        "seed": cfg.seed,
        "deterministic": True,
        "force_row_wise": True,
        "n_jobs": cfg.n_jobs,
        "verbose": -1,
    }
    dtrain = lgb.Dataset(X_train, label=y_train, feature_name=list(feature_names), free_raw_data=False)
    valid_sets = [dtrain]
    valid_names = ["train"]
    callbacks = [lgb.log_evaluation(cfg.verbose_eval)]
    if X_val is not None and len(X_val) and y_val is not None and len(y_val):
        dval = lgb.Dataset(X_val, label=y_val, feature_name=list(feature_names), reference=dtrain)
        valid_sets.append(dval)
        valid_names.append("valid")
        callbacks.append(lgb.early_stopping(cfg.early_stopping_rounds, verbose=True))

    booster = lgb.train(
        params,
        dtrain,
        num_boost_round=cfg.n_estimators,
        valid_sets=valid_sets,
        valid_names=valid_names,
        callbacks=callbacks,
    )
    if model_dir:
        ensure_dir(model_dir)
        booster.save_model(os.path.join(model_dir, "lgbm_pair_model.txt"))
        LOG.info("saved model to %s", os.path.join(model_dir, "lgbm_pair_model.txt"))
    return booster


def feature_importance_table(booster, top_k: int = 40) -> List[Tuple[str, float, float]]:
    names = booster.feature_name()
    gain = booster.feature_importance(importance_type="gain")
    split = booster.feature_importance(importance_type="split")
    order = np.argsort(-gain)
    return [(names[i], float(gain[i]), float(split[i])) for i in order[:top_k]]
