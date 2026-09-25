"""Evaluation: the official macro-averaged F0.5 metric.

F0.5 = (1.25 * P * R) / (0.25 * P + R)

Scoring is done *per Source-1 entity* and then macro-averaged. The edge cases
matter enormously:

* gt empty and pred empty            -> 1.0
* gt empty and pred non-empty        -> 0.0
* gt non-empty and pred empty        -> 0.0
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set

import numpy as np


def entity_f05(gt: Set[int], pred: Set[int]) -> float:
    tp = len(gt & pred)
    if tp == 0:
        return 1.0 if (not gt and not pred) else 0.0
    fp = len(pred - gt)
    fn = len(gt - pred)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    denom = 0.25 * precision + recall
    if denom == 0:
        return 0.0
    return (1.25 * precision * recall) / denom


@dataclass
class EvalResult:
    macro_f05: float
    precision: float
    recall: float
    n_entities: int
    n_entities_with_gt: int
    singleton_score: float           # macro F0.5 restricted to true singletons
    match_score: float               # macro F0.5 restricted to non-singletons
    tp: int
    fp: int
    fn: int
    n_predicted_pairs: int
    by_source: Dict[str, Dict[str, float]] = field(default_factory=dict)
    by_country: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def evaluate(
    predictions: Dict[int, Set[int]],
    ground_truth: Dict[int, Set[int]],
    n_entities: int,
    s1_country: Optional[Sequence[str]] = None,
    cs_order: Optional[Sequence[str]] = None,
    cs_source_of: Optional[np.ndarray] = None,
) -> EvalResult:
    """Macro-averaged F0.5 over the union of evaluated entity indices."""
    keys = set(predictions) | set(ground_truth)
    scores: List[float] = []
    sing_scores: List[float] = []
    match_scores: List[float] = []

    tp_total = fp_total = fn_total = 0
    n_pred_pairs = 0
    # per-source accounting: count tp/fp/fn per candidate source and the macro
    # score restricted to entities whose gt contains that source
    src_stats: Dict[str, List[int]] = {}

    country_scores: Dict[str, List[float]] = {}

    for k in keys:
        gt = ground_truth.get(k, set())
        pred = predictions.get(k, set())
        sc = entity_f05(gt, pred)
        scores.append(sc)
        if not gt:
            sing_scores.append(sc)
        else:
            match_scores.append(sc)

        tp = len(gt & pred)
        fp = len(pred - gt)
        fn = len(gt - pred)
        tp_total += tp
        fp_total += fp
        fn_total += fn
        n_pred_pairs += len(pred)

        if s1_country is not None and k < len(s1_country):
            c = s1_country[k] or "(empty)"
            country_scores.setdefault(c, []).append(sc)

        if cs_order is not None and cs_source_of is not None:
            for s in cs_order:
                st = src_stats.setdefault(s, [0, 0, 0, 0])  # tp,fp,fn,entity_hits
                gt_s = {g for g in gt if cs_order[int(cs_source_of[g])] == s}
                pred_s = {g for g in pred if cs_order[int(cs_source_of[g])] == s}
                t = len(gt_s & pred_s)
                st[0] += t
                st[1] += len(pred_s - gt_s)
                st[2] += len(gt_s - pred_s)
                if gt_s:
                    st[3] += 1

    macro = float(np.mean(scores)) if scores else 0.0
    precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) else 0.0
    recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) else 0.0

    by_source: Dict[str, Dict[str, float]] = {}
    for s, (tp, fp, fn, _hits) in src_stats.items():
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f = (1.25 * p * r) / (0.25 * p + r) if (0.25 * p + r) else 0.0
        by_source[s] = {"precision": p, "recall": r, "f05": f, "tp": tp, "fp": fp, "fn": fn}

    by_country = {
        c: {"macro_f05": float(np.mean(v)), "n": len(v)}
        for c, v in sorted(country_scores.items(), key=lambda kv: -len(kv[1]))
    }

    return EvalResult(
        macro_f05=macro,
        precision=precision,
        recall=recall,
        n_entities=len(keys),
        n_entities_with_gt=sum(1 for k in keys if ground_truth.get(k)),
        singleton_score=float(np.mean(sing_scores)) if sing_scores else 0.0,
        match_score=float(np.mean(match_scores)) if match_scores else 0.0,
        tp=tp_total,
        fp=fp_total,
        fn=fn_total,
        n_predicted_pairs=n_pred_pairs,
        by_source=by_source,
        by_country=by_country,
    )
