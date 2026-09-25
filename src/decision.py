"""Entity-level decision logic and threshold optimisation.

The classifier emits per-pair probabilities; the *metric* is per Source-1 entity.
This module converts probabilities into an accepted match set per entity and
optimises the operating point directly against validation macro F0.5.

Design choices that matter here:

* No forcing. Every Source-1 entity is allowed zero, one or many matches. The
  singleton case (accept nothing) is the default failure mode and is handled by a
  plain probability floor.
* Ambiguity awareness. When the top and second scores are close we can either
  keep both (recall) or trust only the top (precision); the correct behaviour is
  selected empirically via ``top_only`` / ``margin`` knobs.
* Source-specific calibration. S1->S2 and S1->S3 may not share an operating
  point; a per-source threshold is retained only if it helps validation.
* High-confidence override, so a near-certain pair is never lost to a low floor.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from .candidates import CandidatePair
from .evaluation import EvalResult, evaluate
from .utils import LOG


@dataclass
class DecisionConfig:
    threshold: float = 0.5
    threshold_s2: Optional[float] = None        # None -> use global
    threshold_s3: Optional[float] = None
    high_conf: float = 0.995                     # always accept at/above this
    margin: float = 0.0                          # if top-second < margin, cut to top-1
    top_only: bool = False                       # keep only argmax per entity
    max_matches_per_entity: int = 0              # 0 -> unlimited
    require_candidate: bool = True               # empty candidate set -> empty pred

    def threshold_for_source(self, src: str) -> float:
        if src == "S2" and self.threshold_s2 is not None:
            return self.threshold_s2
        if src == "S3" and self.threshold_s3 is not None:
            return self.threshold_s3
        return self.threshold


def group_pairs_by_entity(pairs: Sequence[CandidatePair]) -> Dict[int, List[int]]:
    """s1_index -> list of row indices (into ``pairs``/probs)."""
    g: Dict[int, List[int]] = {}
    for r, p in enumerate(pairs):
        g.setdefault(p.s1_index, []).append(r)
    return g


def _accepted_for_entity(
    rows: List[int],
    probs: np.ndarray,
    pairs: Sequence[CandidatePair],
    cs_order: Sequence[str],
    cs_source_of: np.ndarray,
    cfg: DecisionConfig,
) -> Set[int]:
    if not rows:
        return set()

    scored: List[Tuple[float, int, str]] = []
    for r in rows:
        p = float(probs[r])
        g = pairs[r].gid
        src = cs_order[int(cs_source_of[g])]
        scored.append((p, g, src))
    scored.sort(key=lambda x: -x[0])

    if cfg.top_only:
        top_p, top_g, _ = scored[0]
        thr = cfg.threshold_for_source(scored[0][2])
        if top_p >= thr or top_p >= cfg.high_conf:
            return {top_g}
        return set()

    accepted: List[Tuple[float, int]] = []
    for p, g, src in scored:
        thr = cfg.threshold_for_source(src)
        if p >= thr or p >= cfg.high_conf:
            accepted.append((p, g))

    # ambiguity: near-ties -> optionally trust only the top candidate
    if cfg.margin and cfg.margin > 0 and len(accepted) >= 2:
        if (accepted[0][0] - accepted[1][0]) < cfg.margin:
            accepted = accepted[:1]

    if cfg.max_matches_per_entity and len(accepted) > cfg.max_matches_per_entity:
        accepted = accepted[: cfg.max_matches_per_entity]

    return {g for _, g in accepted}


def apply_decision(
    pairs: Sequence[CandidatePair],
    probs: np.ndarray,
    n_s1: int,
    cs_order: Sequence[str],
    cs_source_of: np.ndarray,
    cfg: DecisionConfig,
    groups: Optional[Dict[int, List[int]]] = None,
) -> Dict[int, Set[int]]:
    if groups is None:
        groups = group_pairs_by_entity(pairs)
    preds: Dict[int, Set[int]] = {}
    for i in range(n_s1):
        rows = groups.get(i)
        if not rows:
            preds[i] = set()
            continue
        preds[i] = _accepted_for_entity(rows, probs, pairs, cs_order, cs_source_of, cfg)
    return preds


# --------------------------------------------------------------------------- #
# Threshold optimisation
# --------------------------------------------------------------------------- #
DEFAULT_GRID: List[float] = [
    0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75,
    0.80, 0.85, 0.875, 0.90, 0.925, 0.95, 0.975,
]


@dataclass
class ThresholdSearchResult:
    best: DecisionConfig
    best_score: float
    table: List[dict] = field(default_factory=list)


def _fine_grid(center: float, lo: float = 0.02, hi: float = 0.995, step: float = 0.005) -> List[float]:
    start = max(lo, center - 0.10)
    end = min(hi, center + 0.10)
    vals = np.arange(start, end + 1e-9, step)
    return [round(float(v), 4) for v in vals]


def optimize_threshold(
    pairs: Sequence[CandidatePair],
    probs: np.ndarray,
    ground_truth: Dict[int, Set[int]],
    n_s1: int,
    cs_order: Sequence[str],
    cs_source_of: np.ndarray,
    s1_country: Optional[Sequence[str]] = None,
    coarse_grid: Sequence[float] = tuple(DEFAULT_GRID),
    fine: bool = True,
) -> ThresholdSearchResult:
    groups = group_pairs_by_entity(pairs)
    table: List[dict] = []
    best_cfg = DecisionConfig(threshold=0.5)
    best = -1.0

    def score_cfg(cfg: DecisionConfig) -> float:
        preds = apply_decision(pairs, probs, n_s1, cs_order, cs_source_of, cfg, groups)
        res = evaluate(preds, ground_truth, n_s1, s1_country=s1_country,
                       cs_order=cs_order, cs_source_of=cs_source_of)
        return res.macro_f05

    LOG.info("threshold search: coarse grid over %d values", len(coarse_grid))
    for t in coarse_grid:
        cfg = DecisionConfig(threshold=float(t))
        sc = score_cfg(cfg)
        table.append({"threshold": float(t), "macro_f05": sc})
        if sc > best:
            best, best_cfg = sc, cfg

    if fine:
        LOG.info("threshold search: fine grid around %.3f", best_cfg.threshold)
        for t in _fine_grid(best_cfg.threshold):
            cfg = DecisionConfig(threshold=t)
            sc = score_cfg(cfg)
            table.append({"threshold": t, "macro_f05": sc})
            if sc > best:
                best, best_cfg = sc, cfg

    LOG.info("best threshold=%.4f macro_f05=%.5f", best_cfg.threshold, best)
    return ThresholdSearchResult(best=best_cfg, best_score=best, table=table)


def optimize_source_thresholds(
    pairs: Sequence[CandidatePair],
    probs: np.ndarray,
    ground_truth: Dict[int, Set[int]],
    n_s1: int,
    cs_order: Sequence[str],
    cs_source_of: np.ndarray,
    base: DecisionConfig,
    s1_country: Optional[Sequence[str]] = None,
    grid: Sequence[float] = tuple(DEFAULT_GRID),
) -> Tuple[DecisionConfig, float, List[dict]]:
    """Try independent S2/S3 thresholds on top of the global one; keep if better."""
    groups = group_pairs_by_entity(pairs)

    def score_cfg(cfg: DecisionConfig) -> float:
        preds = apply_decision(pairs, probs, n_s1, cs_order, cs_source_of, cfg, groups)
        return evaluate(preds, ground_truth, n_s1, s1_country=s1_country,
                        cs_order=cs_order, cs_source_of=cs_source_of).macro_f05

    best_cfg = base
    best_sc = score_cfg(base)
    table = [{"stage": "global", "threshold": base.threshold, "macro_f05": best_sc}]
    LOG.info("source-specific search: baseline macro_f05=%.5f", best_sc)

    for src, attr in (("S2", "threshold_s2"), ("S3", "threshold_s3")):
        cur = getattr(best_cfg, attr)
        local_best_sc = best_sc
        local_best_t = cur if cur is not None else best_cfg.threshold
        for t in grid:
            trial = DecisionConfig(**{**best_cfg.__dict__, attr: float(t)})
            sc = score_cfg(trial)
            table.append({"stage": src, "threshold": float(t), "macro_f05": sc})
            if sc > local_best_sc + 1e-6:
                local_best_sc = sc
                local_best_t = float(t)
        if local_best_sc > best_sc + 1e-6:
            LOG.info("  %s threshold -> %.4f improves to %.5f", src, local_best_t, local_best_sc)
            best_cfg = DecisionConfig(**{**best_cfg.__dict__, attr: local_best_t})
            best_sc = local_best_sc
        else:
            LOG.info("  %s threshold kept at global (no improvement)", src)
    return best_cfg, best_sc, table


def optimize_ambiguity(
    pairs: Sequence[CandidatePair],
    probs: np.ndarray,
    ground_truth: Dict[int, Set[int]],
    n_s1: int,
    cs_order: Sequence[str],
    cs_source_of: np.ndarray,
    base: DecisionConfig,
    s1_country: Optional[Sequence[str]] = None,
) -> Tuple[DecisionConfig, float, List[dict]]:
    """Search margin / top_only / max_matches variants on top of ``base``."""
    groups = group_pairs_by_entity(pairs)

    def score_cfg(cfg: DecisionConfig) -> float:
        preds = apply_decision(pairs, probs, n_s1, cs_order, cs_source_of, cfg, groups)
        return evaluate(preds, ground_truth, n_s1, s1_country=s1_country,
                        cs_order=cs_order, cs_source_of=cs_source_of).macro_f05

    best_cfg, best_sc = base, score_cfg(base)
    table = [{"variant": "base", "macro_f05": best_sc}]

    variants: List[DecisionConfig] = []
    for top_only in (True,):
        variants.append(DecisionConfig(**{**base.__dict__, "top_only": top_only}))
    for m in (0.05, 0.10, 0.20, 0.35):
        variants.append(DecisionConfig(**{**base.__dict__, "margin": m}))
    for k in (1, 2, 3, 5):
        variants.append(DecisionConfig(**{**base.__dict__, "max_matches_per_entity": k}))

    for cfg in variants:
        sc = score_cfg(cfg)
        label = f"top_only={cfg.top_only},margin={cfg.margin},max={cfg.max_matches_per_entity}"
        table.append({"variant": label, "macro_f05": sc})
        if sc > best_sc + 1e-6:
            best_sc, best_cfg = sc, cfg
            LOG.info("  ambiguity variant %s improves to %.5f", label, sc)
    return best_cfg, best_sc, table
