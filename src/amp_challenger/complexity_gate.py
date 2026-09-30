# -*- coding: utf-8 -*-
"""Length-robust sequence-complexity gate.

Replaces the ``len(set(seq)) / len(seq) >= 0.45`` guard that the competition
selector and :func:`amp_scoring.composition_realism_factor` used to share.

Why the old guard had to go
---------------------------
On the effective 17-letter AMP alphabet (C/D/E forbidden) the distinct-residue
ratio is bounded above by ``min(n, 17) / n``, so a 29-mer had to spend almost
the whole alphabet to clear 0.45 while a 10-mer needed five residues. Measured
on the deployed 50k library it passed 67.6 % of 10-14mers and 0.3 % of
25-29mers, and on the project's own reference corpus
(``combined_amp_databases_unique.fasta``, 22,180 validated AMPs) it rejected
49.7 % of real AMPs -- 94.7 % of 35-39mers and 100 % of those 45 aa or longer.
It was a length filter, and it was silently steering the top-100 short: the
deployed top-100 topped out at 24 aa while the library reached 29, and all
6,149 library peptides of 25 aa or more produced zero selected candidates.

The replacement
---------------
Complexity is normalised Shannon entropy, ``H(seq) / log2(min(len(seq), 20))``.
The denominator is the entropy of a maximally diverse sequence *of the same
length*, which puts 8-mers and 40-mers on one scale. The threshold is then the
5th percentile of real AMPs **of that length** (see
``scripts/build_complexity_gate.py``), so retention is flat in length by
construction: 94.5-95.6 % in every length bin, against the old 0.3-67.6 %.

A sequence must also be free of dispersed tandem repeats
(:func:`compliance.has_low_complexity_repeat`); entropy alone accepts
``GLMQFIKRGLMQFIKR``, which is a generator artifact rather than a peptide.
"""
from __future__ import annotations

import json
import logging
import math
from collections import Counter
from pathlib import Path

from .compliance import has_low_complexity_repeat

logger = logging.getLogger(__name__)

MIN_GATE_LEN: int = 5
MAX_GATE_LEN: int = 50

DEFAULT_GATE_PATH: Path = (
    Path(__file__).resolve().parents[2] / "data" / "external" / "complexity_gate.json"
)

# Fallback table (percentile 5, calibrated 2026-09-22 on the 22,180-sequence
# combined_amp_databases_unique.fasta corpus). Embedded so a missing JSON file
# degrades to the calibrated gate rather than to no gate at all -- the failure
# mode that let the novelty reference silently shrink back to its 200-sequence
# panel when all_amps_indep_nr80.fasta went missing.
_FALLBACK_THRESHOLDS: dict[int, float] = {
    5: 0.3333, 6: 0.3333, 7: 0.3333, 8: 0.3188, 9: 0.308, 10: 0.2927, 11:
    0.3152, 12: 0.3542, 13: 0.3718, 14: 0.3911, 15: 0.37, 16: 0.4032, 17:
    0.3873, 18: 0.431, 19: 0.4517, 20: 0.4411, 21: 0.3997, 22: 0.3936, 23:
    0.4307, 24: 0.4584, 25: 0.4483, 26: 0.4692, 27: 0.4366, 28: 0.4798, 29:
    0.5066, 30: 0.5909, 31: 0.5643, 32: 0.5822, 33: 0.5953, 34: 0.6419, 35:
    0.6354, 36: 0.6584, 37: 0.6375, 38: 0.6673, 39: 0.6797, 40: 0.7287, 41:
    0.7275, 42: 0.7203, 43: 0.7152, 44: 0.7289, 45: 0.7421, 46: 0.7496, 47:
    0.7339, 48: 0.7086, 49: 0.6882, 50: 0.6808
}


def entropy_ratio(sequence: str, alphabet_size: int = 20) -> float:
    """Shannon entropy normalised by the maximum achievable at this length.

    Returns a value in ``[0, 1]`` that is comparable across peptide lengths:
    1.0 means every residue is distinct (or the composition is uniform over the
    full alphabet once the peptide is longer than it), 0.0 means a homopolymer.
    """
    seq = sequence.upper().strip()
    n = len(seq)
    if n < 2:
        return 0.0
    counts = Counter(seq)
    entropy = -sum((c / n) * math.log2(c / n) for c in counts.values())
    max_entropy = math.log2(min(n, alphabet_size))
    if max_entropy <= 0.0:
        return 0.0
    return max(0.0, min(1.0, entropy / max_entropy))


class ComplexityGate:
    """Per-length complexity thresholds calibrated on real AMPs."""

    def __init__(self, path: str | Path | None = None) -> None:
        self._path = Path(path) if path is not None else DEFAULT_GATE_PATH
        self._thresholds: dict[int, float] = {}
        self._metadata: dict[str, object] = {}
        self._loaded = False
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            self._thresholds = {
                int(k): float(v) for k, v in data["thresholds"].items()
            }
            self._metadata = {
                k: v for k, v in data.items() if k != "thresholds"
            }
            self._loaded = True
        except FileNotFoundError:
            logger.warning(
                "ComplexityGate: table not found at %s -- using the embedded "
                "calibration. Regenerate with scripts/build_complexity_gate.py.",
                self._path,
            )
            self._thresholds = dict(_FALLBACK_THRESHOLDS)
        except (KeyError, ValueError, json.JSONDecodeError) as exc:
            logger.warning(
                "ComplexityGate: could not parse %s (%s) -- using the embedded "
                "calibration.", self._path, exc,
            )
            self._thresholds = dict(_FALLBACK_THRESHOLDS)

    @property
    def is_calibrated(self) -> bool:
        """True when the on-disk calibration table was loaded."""
        return self._loaded

    @property
    def path(self) -> str:
        return str(self._path)

    @property
    def metadata(self) -> dict[str, object]:
        return dict(self._metadata)

    def threshold(self, length: int) -> float:
        """Complexity threshold for a peptide of ``length`` residues.

        Lengths outside the calibrated range clamp to the nearest calibrated
        end rather than falling open, so an out-of-range peptide is still held
        to the closest evidence-backed bar.
        """
        if not self._thresholds:
            return 0.0
        clamped = max(MIN_GATE_LEN, min(MAX_GATE_LEN, length))
        if clamped in self._thresholds:
            return self._thresholds[clamped]
        nearest = min(self._thresholds, key=lambda k: abs(k - clamped))
        return self._thresholds[nearest]

    def score(self, sequence: str) -> float:
        """Length-normalised complexity of ``sequence`` (higher = more complex)."""
        return entropy_ratio(sequence)

    def passes(self, sequence: str) -> bool:
        """True when the sequence clears its length's bar and has no tandem repeat."""
        seq = sequence.upper().strip()
        if not seq:
            return False
        if has_low_complexity_repeat(seq):
            return False
        return entropy_ratio(seq) >= self.threshold(len(seq))


_gate: ComplexityGate | None = None
_gate_path: str | None = None


def get_complexity_gate(path: str | Path | None = None) -> ComplexityGate:
    """Return (or lazily create) the module-level ComplexityGate singleton."""
    global _gate, _gate_path
    key = str(path) if path is not None else None
    if _gate is None or _gate_path != key:
        _gate = ComplexityGate(path)
        _gate_path = key
    return _gate


__all__ = [
    "ComplexityGate",
    "DEFAULT_GATE_PATH",
    "MAX_GATE_LEN",
    "MIN_GATE_LEN",
    "entropy_ratio",
    "get_complexity_gate",
]
