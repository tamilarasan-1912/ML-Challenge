"""Probability calibration.

Raw LightGBM scores are already reasonably calibrated, but we still evaluate
isotonic and Platt scaling and retain a calibrator only when it improves
validation macro F0.5 *after* the decision layer is applied. Selection is done by
the caller (``decision`` module) using the accept/reject matrix, so here we only
fit and expose the transforms.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .utils import LOG


@dataclass
class Calibrator:
    kind: str = "none"          # none | isotonic | sigmoid
    model: object = None

    def transform(self, p: np.ndarray) -> np.ndarray:
        p = np.asarray(p, dtype=np.float64)
        if self.kind == "none" or self.model is None:
            return p
        if self.kind == "isotonic":
            return np.clip(self.model.predict(p), 0.0, 1.0)
        # sigmoid (Platt): logistic regression fitted on logit(p)
        eps = 1e-6
        pc = np.clip(p, eps, 1 - eps)
        z = np.log(pc / (1 - pc)).reshape(-1, 1)
        return self.model.predict_proba(z)[:, 1]

    def as_dict(self) -> dict:
        return {"kind": self.kind}


def fit_calibrators(y: np.ndarray, p: np.ndarray) -> dict:
    """Fit candidate calibrators; return {'none':..., 'isotonic':..., 'sigmoid':...}."""
    out = {"none": Calibrator("none")}
    p = np.asarray(p, dtype=np.float64)
    try:
        from sklearn.isotonic import IsotonicRegression

        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(p, y)
        out["isotonic"] = Calibrator("isotonic", iso)
    except Exception as exc:  # pragma: no cover
        LOG.warning("isotonic calibration unavailable: %s", exc)

    try:
        from sklearn.linear_model import LogisticRegression

        eps = 1e-6
        pc = np.clip(p, eps, 1 - eps)
        z = np.log(pc / (1 - pc)).reshape(-1, 1)
        lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
        lr.fit(z, y)
        out["sigmoid"] = Calibrator("sigmoid", lr)
    except Exception as exc:  # pragma: no cover
        LOG.warning("platt calibration unavailable: %s", exc)
    return out
