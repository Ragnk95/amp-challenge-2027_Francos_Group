"""Optional DBAASP-trained MIC regressor.

The production pipeline already exposes raw and calibrated APEX MIC columns.
This module adds a separate, lightweight regressor surface for experimental
MIC models trained from DBAASP-style exports. It intentionally has no sklearn
runtime dependency: training writes plain JSON coefficients, inference reads
that JSON and evaluates a standardised linear model.
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .amp_scoring import (
    compute_amphipathicity,
    compute_complexity_score,
    compute_helix_propensity,
    compute_hydrophobicity,
    compute_membrane_insertion_score,
    compute_net_charge,
    compute_stability_proxy,
)

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = _REPO_ROOT / "data" / "external" / "dbaasp_mic_regressor.json"

VALID_AA = frozenset("ACDEFGHIKLMNPQRSTVWY")

MIC_FEATURE_NAMES: list[str] = [
    "length_norm",
    "net_charge_norm",
    "kr_fraction",
    "hydrophobicity",
    "hydrophobic_fraction",
    "amphipathicity",
    "membrane_insertion_score",
    "helix_propensity",
    "stability_proxy",
    "complexity_score",
    "trp_fraction",
    "phe_fraction",
    "leu_fraction",
    "gly_fraction",
    "pro_fraction",
]


DBAASP_MIC_TARGETS: tuple[str, ...] = (
    "gram_neg",
    "gram_pos",
    "ecoli_atcc11775",
    "ecoli_k12",
    "ecoli_aic221",
    "ecoli_cre",
    "paeruginosa_pao1",
    "paeruginosa_pa14",
    "paeruginosa_baa3197",
    "kpneumoniae_13883",
    "kpneumoniae_baa2342",
    "abaumannii_19606",
    "abaumannii_baa1605",
    "ecloacae",
    "ecloacae_13047",
    "senterica",
    "senterica_typhimurium",
    "bsubtilis",
    "bsubtilis_23857",
    "saureus_atcc12600",
    "saureus_mrsa",
    "vre_faecalis",
    "vre_faecium",
    "vre_faecium_700221",
)


@dataclass(frozen=True, slots=True)
class MICPrediction:
    """Predicted MIC for one target."""

    target: str
    mic_ugml: float
    log10_mic_ugml: float
    fold_uncertainty: float
    n_train: int


@dataclass(frozen=True, slots=True)
class _TargetModel:
    coef: tuple[float, ...]
    intercept: float
    x_mean: tuple[float, ...]
    x_scale: tuple[float, ...]
    residual_std_log10: float
    n_train: int


def sanitize_sequence(sequence: str) -> str:
    """Upper-case and validate a canonical peptide sequence."""
    seq = "".join(ch for ch in sequence.upper().strip() if ch.isalpha())
    if not seq or any(ch not in VALID_AA for ch in seq):
        return ""
    return seq


def extract_mic_features(sequence: str) -> list[float]:
    """Feature vector consumed by the DBAASP MIC regressor.

    The features are deliberately cheap and sequence-only so the model can be
    used during fast ranking without loading ESM/APEX. Feature order is fixed
    by :data:`MIC_FEATURE_NAMES` and stored in the trained JSON.
    """
    seq = sanitize_sequence(sequence)
    if not seq:
        return [0.0] * len(MIC_FEATURE_NAMES)
    n = len(seq)
    hydrophobic = sum(aa in "ILVFMWAP" for aa in seq) / n
    return [
        min(n / 50.0, 1.0),
        max(-1.0, min(1.0, compute_net_charge(seq) / 12.0)),
        (seq.count("K") + seq.count("R")) / n,
        compute_hydrophobicity(seq),
        hydrophobic,
        compute_amphipathicity(seq),
        compute_membrane_insertion_score(seq),
        compute_helix_propensity(seq),
        compute_stability_proxy(seq),
        compute_complexity_score(seq),
        seq.count("W") / n,
        seq.count("F") / n,
        seq.count("L") / n,
        seq.count("G") / n,
        seq.count("P") / n,
    ]


class DBAASPMICRegressor:
    """Load and evaluate a trained DBAASP MIC regressor JSON.

    Missing or malformed models are treated as an optional-component miss:
    :attr:`is_loaded` is false and prediction methods return ``None``/empty
    results instead of raising in the main pipeline.
    """

    __slots__ = ("_loaded", "_path", "_targets")

    def __init__(self, path: Path | None = None) -> None:
        self._path = path or DEFAULT_MODEL_PATH
        self._loaded = False
        self._targets: dict[str, _TargetModel] = {}
        self._load()

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def targets(self) -> list[str]:
        return sorted(self._targets.keys())

    def _load(self) -> None:
        if not self._path.exists():
            logger.info(
                "DBAASPMICRegressor: %s not found; DBAASP MIC predictions disabled.",
                self._path,
            )
            return
        try:
            with self._path.open(encoding="utf-8") as fh:
                payload: dict[str, Any] = json.load(fh)
            features = payload.get("features")
            if features != MIC_FEATURE_NAMES:
                raise ValueError("feature schema mismatch")
            raw_targets = payload.get("targets")
            if not isinstance(raw_targets, dict):
                raise ValueError("missing targets block")
            for target, params in raw_targets.items():
                if not isinstance(target, str) or not isinstance(params, dict):
                    continue
                model = _TargetModel(
                    coef=tuple(float(v) for v in params["coef"]),
                    intercept=float(params["intercept"]),
                    x_mean=tuple(float(v) for v in params["x_mean"]),
                    x_scale=tuple(float(v) for v in params["x_scale"]),
                    residual_std_log10=float(params.get("residual_std_log10", 0.5)),
                    n_train=int(params.get("n_train", 0)),
                )
                if (
                    len(model.coef)
                    == len(model.x_mean)
                    == len(model.x_scale)
                    == len(MIC_FEATURE_NAMES)
                ):
                    self._targets[target] = model
        except Exception as exc:
            logger.warning("DBAASPMICRegressor: failed to load %s: %s", self._path, exc)
            return
        self._loaded = bool(self._targets)
        if self._loaded:
            logger.info(
                "DBAASPMICRegressor loaded %d target models from %s",
                len(self._targets),
                self._path.name,
            )

    def predict_log10(self, sequence: str, target: str) -> float | None:
        """Return log10(MIC µg/mL), or ``None`` when unavailable."""
        model = self._targets.get(target)
        if model is None:
            return None
        features = extract_mic_features(sequence)
        z = [
            (value - mean) / scale if scale > 0 else 0.0
            for value, mean, scale in zip(features, model.x_mean, model.x_scale)
        ]
        pred = model.intercept + sum(c * x for c, x in zip(model.coef, z))
        # Keep predictions in a wet-lab-plausible numeric range: 0.01-4096 ug/mL.
        return max(-2.0, min(math.log10(4096.0), float(pred)))

    def predict(self, sequence: str, target: str) -> MICPrediction | None:
        """Return a MIC prediction for one target, if the target model exists."""
        log10_mic = self.predict_log10(sequence, target)
        model = self._targets.get(target)
        if log10_mic is None or model is None:
            return None
        return MICPrediction(
            target=target,
            mic_ugml=10.0 ** log10_mic,
            log10_mic_ugml=log10_mic,
            fold_uncertainty=10.0 ** model.residual_std_log10,
            n_train=model.n_train,
        )

    def predict_many(
        self,
        sequence: str,
        targets: Sequence[str] | None = None,
    ) -> dict[str, MICPrediction]:
        """Predict all requested targets available in the model."""
        requested = list(targets) if targets is not None else self.targets
        out: dict[str, MICPrediction] = {}
        for target in requested:
            pred = self.predict(sequence, target)
            if pred is not None:
                out[target] = pred
        return out


_singleton: DBAASPMICRegressor | None = None


def get_dbaasp_mic_regressor() -> DBAASPMICRegressor:
    """Process-wide singleton matching the optional classifier pattern."""
    global _singleton
    if _singleton is None:
        _singleton = DBAASPMICRegressor()
    return _singleton
