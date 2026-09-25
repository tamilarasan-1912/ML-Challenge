"""End-to-end inference for a split (train validation or test).

Pipeline: load -> normalize -> index -> candidate generation -> features ->
LightGBM -> calibration -> entity-level decision -> optional consistency.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from . import indexing as idx
from .blocking import BlockingConfig, build_blocking_index
from .candidates import CandidatePair, generate_candidates
from .decision import DecisionConfig, apply_decision, group_pairs_by_entity
from .features import FeatureExtractor
from .indexing import CandidateSpace, FrequencyStats, SourceTable
from .utils import LOG, PeakMemory, timed


@dataclass
class InferenceArtifacts:
    s1: SourceTable
    cs: CandidateSpace
    pairs: List[CandidatePair]
    probs: np.ndarray
    preds: Dict[int, Set[int]]
    groups: Dict[int, List[int]]


def build_candidate_space_from_tables(tables: Dict[str, SourceTable]) -> CandidateSpace:
    return idx.build_candidate_space(tables, order=("S2", "S3"))


def run_candidates_and_features(
    s1: SourceTable,
    cs: CandidateSpace,
    stats: FrequencyStats,
    block_cfg: BlockingConfig,
    feature_cfg=None,
    log_every: int = 20000,
) -> Tuple[List[CandidatePair], np.ndarray]:
    with timed("build_blocking_index"):
        bi = build_blocking_index(cs, stats, block_cfg)
    with timed("generate_candidates"):
        pairs = generate_candidates(s1, bi, block_cfg, log_every=log_every)
    LOG.info("generated %d candidate pairs (%.2f per S1)", len(pairs), len(pairs) / max(1, s1.n))
    extractor = FeatureExtractor(s1, cs, stats, bi, rare_df=block_cfg.rare_name_token_df)
    with timed("extract_features"):
        X = extractor.extract(pairs, log_every=max(500000, 1))
    return pairs, X


def predict_probabilities(booster, X: np.ndarray) -> np.ndarray:
    return booster.predict(X, num_iteration=getattr(booster, "best_iteration", None))


def decide(
    pairs: Sequence[CandidatePair],
    probs: np.ndarray,
    n_s1: int,
    cs: CandidateSpace,
    cfg: DecisionConfig,
) -> Tuple[Dict[int, Set[int]], Dict[int, List[int]]]:
    groups = group_pairs_by_entity(pairs)
    preds = apply_decision(pairs, probs, n_s1, cs.order, cs.source_of, cfg, groups)
    return preds, groups
