"""Hard-negative mining and difficulty stratification.

Easy negatives (``Acme Widgets`` vs ``Zenith Logistics``) dominate the candidate
set and would swamp the gradient. We stratify negatives by how deceptively
similar they are to the Source-1 entity and retain the dangerous ones while
down-sampling the trivial ones in the *training* split only. Validation always
keeps the full candidate set so that measured F0.5 reflects deployment.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .candidates import CandidatePair
from .features import FEATURE_INDEX
from .utils import LOG

# feature columns used to judge difficulty
_I_NAME = FEATURE_INDEX["name_ratio"]
_I_CORE_EXACT = FEATURE_INDEX["name_core_exact"]
_I_TSET = FEATURE_INDEX["name_token_set_ratio"]
_I_ADDR = FEATURE_INDEX["addr_ratio"]
_I_ADDR_EXACT = FEATURE_INDEX["addr_exact"]
_I_POSTAL = FEATURE_INDEX["postal_match"]
_I_HOUSE = FEATURE_INDEX["house_number_match"]
_I_OVERLAP = FEATURE_INDEX["token_overlap"]
_I_COUNTRY_EQ = FEATURE_INDEX["country_equal"]


@dataclass
class MiningConfig:
    keep_very_hard: bool = True
    keep_hard: bool = True
    hard_neg_cap_ratio: float = 20.0     # max negatives per positive after mining
    medium_sample_rate: float = 0.35
    easy_sample_rate: float = 0.05
    seed: int = 42


def difficulty_tier(X: np.ndarray, row: int) -> int:
    """0=very_hard, 1=hard, 2=medium, 3=easy."""
    name = X[row, _I_NAME]
    tset = X[row, _I_TSET]
    core_exact = X[row, _I_CORE_EXACT]
    addr = X[row, _I_ADDR]
    postal = X[row, _I_POSTAL]
    house = X[row, _I_HOUSE]
    overlap = X[row, _I_OVERLAP]

    if (name >= 0.85 and addr >= 0.85) or core_exact > 0 or (name >= 0.90 and (postal or house)):
        return 0
    if name >= 0.72 or (name >= 0.55 and addr >= 0.70) or (name >= 0.60 and house) or tset >= 0.80:
        return 1
    if name >= 0.45 or overlap >= 0.5 or tset >= 0.60 or addr >= 0.75:
        return 2
    return 3


def mine_hard_negatives(
    X: np.ndarray,
    y: np.ndarray,
    cfg: MiningConfig,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """Return indices (into X/y) selected for training."""
    rng = np.random.default_rng(cfg.seed)
    n = len(y)
    tiers = np.empty(n, dtype=np.int8)
    for r in range(n):
        tiers[r] = 0 if y[r] == 1 else difficulty_tier(X, r)

    pos_idx = np.nonzero(y == 1)[0]
    counts = {t: int(np.sum((y == 0) & (tiers == t))) for t in range(4)}
    n_pos = max(1, len(pos_idx))
    cap = int(cfg.hard_neg_cap_ratio * n_pos)

    chosen: List[np.ndarray] = [pos_idx]
    budget = cap

    def take(tier: int, rate: float) -> None:
        nonlocal budget
        if budget <= 0:
            return
        pool = np.nonzero((y == 0) & (tiers == tier))[0]
        if pool.size == 0:
            return
        if rate < 1.0:
            k = max(1, int(pool.size * rate))
            pool = rng.choice(pool, size=k, replace=False)
        if pool.size > budget:
            pool = rng.choice(pool, size=budget, replace=False)
        chosen.append(pool)
        budget -= pool.size

    if cfg.keep_very_hard:
        take(0, 1.0)
    if cfg.keep_hard:
        take(1, 1.0)
    take(2, cfg.medium_sample_rate)
    take(3, cfg.easy_sample_rate)

    sel = np.concatenate(chosen)
    sel = np.unique(sel)
    info = {
        "n_total": n,
        "n_pos": int(len(pos_idx)),
        "n_neg_total": int(n - len(pos_idx)),
        "tier0_very_hard": counts[0],
        "tier1_hard": counts[1],
        "tier2_medium": counts[2],
        "tier3_easy": counts[3],
        "n_selected": int(len(sel)),
        "n_selected_neg": int(len(sel) - len(pos_idx)),
    }
    LOG.info(
        "hard-negative mining: pos=%d tiers(vh/h/m/e)=%d/%d/%d/%d selected=%d",
        info["n_pos"], counts[0], counts[1], counts[2], counts[3], info["n_selected"],
    )
    return sel, info
