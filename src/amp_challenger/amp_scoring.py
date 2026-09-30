"""AMP-specific scoring functions.

Provides peptide-level scores for:
  - Antimicrobial activity proxy  (charge + hydrophobicity + amphipathicity)
  - Hemolysis risk proxy           (charge inversion at cytoplasmic pH, Trp content)
  - Selectivity Index estimate    (AMP activity / hemolysis risk)
  - Membrane-disruption potential  (Wimley–White ΔG insertion proxy)
  - Protease stability proxy       (cleavage site count heuristic)
  - Structural propensity          (helix / sheet / amphipathicity prediction)

All scores are normalised to [0, 1].  No ML model is required at scoring
time; the heavier ESMC plausibility is delegated to the shared
``ESMCPeptideScorer`` in the main pipeline.
"""
from __future__ import annotations

import functools
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .complexity_gate import ComplexityGate, get_complexity_gate


# ---------------------------------------------------------------------------
# Amino-acid property tables
# ---------------------------------------------------------------------------

# Kyte-Doolittle hydrophobicity (normalised to [0, 1])
# Negative values indicate hydrophilic residues; positive = hydrophobic.
_KD: dict[str, float] = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5,
    "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,  "L": 3.8,  "K": -3.9,
    "M": 1.9,  "F": 2.8,  "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9,
    "Y": -1.3, "V": 4.2,
}
_KD_MIN, _KD_MAX = -4.5, 4.5
_KD_NORM: dict[str, float] = {aa: (v - _KD_MIN) / (_KD_MAX - _KD_MIN) for aa, v in _KD.items()}

# Wimley–White whole-residue ΔG of insertion into POPC bilayer (kcal/mol)
# Negative = favours insertion; positive = disfavours.
_WW: dict[str, float] = {
    "A": -0.17, "R": 0.81,  "N": 0.42,  "D": 1.23,  "C": -0.24, "Q": 0.58,
    "E": 2.02,  "G": -0.01, "H": 0.96,  "I": -0.31, "L": -0.56, "K": 0.99,
    "M": -0.23, "F": -1.13, "P": 0.45,  "S": 0.13,  "T": 0.14,  "W": -1.85,
    "Y": -0.94, "V": -0.07,
}

# Formal charge contribution at pH 7 (simplified)
_CHARGE: dict[str, float] = {
    "R": +1.0, "K": +1.0, "H": +0.1,
    "D": -1.0, "E": -1.0,
}

# Residues prone to chymotrypsin cleavage (C-terminal side).
# K and R intentionally excluded: they are essential cationic residues for
# bacterial membrane targeting and their removal would penalise the very
# charge property that drives AMP–LPS/LTA electrostatic interaction.
# Eisenberg consensus hydrophobicity (Eisenberg et al., 1984). This is the scale
# the hydrophobic moment is defined on. It is NOT Kyte-Doolittle: the two differ
# in sign for the aromatics (W +0.81 here vs -0.90 in KD, Y +0.26 vs -1.30),
# which decides whether tryptophan lands on the apolar face or the polar one.
_EISENBERG: dict[str, float] = {
    "A": 0.62, "R": -2.53, "N": -0.78, "D": -0.90, "C": 0.29,
    "Q": -0.85, "E": -0.74, "G": 0.48, "H": -0.40, "I": 1.38,
    "L": 1.06, "K": -1.50, "M": 0.64, "F": 1.19, "P": 0.12,
    "S": -0.18, "T": -0.05, "W": 0.81, "Y": 0.26, "V": 1.08,
}
# Largest absolute value on the scale (arginine). The mean per-residue moment
# cannot exceed it, so it is the normaliser, exactly as 4.5 was for KD.
_EISENBERG_MAX: float = 2.53

_CLEAVAGE_SITES: frozenset[str] = frozenset("FYW")


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=8192)
def compute_net_charge(sequence: str) -> float:
    """Formal net charge at pH 7 from side chains only.

    The free N-terminus (+1) and C-terminus (-1) cancel for an unmodified
    zwitterion, so neither is added. Note this means C-terminal amidation --
    common in natural AMPs and worth +1 -- is not modelled.
    """
    charge = 1.0 - 1.0  # N-term and C-term cancel for zwitterion
    for aa in sequence.upper():
        charge += _CHARGE.get(aa, 0.0)
    return charge


# Half-width of the charge bell. Was 6.0, which after the 0.35*L peak
# left almost the whole library within one unit of its peak: the term
# excluded anions but separated nothing at the top.
_BELL_WIDTH: float = 3.0

def normalize_charge(charge: float, mode: str = "bell",
                     length: int | None = None) -> float:
    """Normalise net charge to [0, 1].

    Parameters
    ----------
    charge:
        Net formal charge at pH 7.
    mode:
        ``"bell"`` (default): peaks at ``0.35 x length`` when ``length``
        is given, otherwise at +5, and decays symmetrically.
        Formula: ``max(0, 1 − |charge − peak| / 6)``.
        Biologically justified: optimal bacterial-membrane interaction at
        +4 to +6 for most helical AMPs (Shai 2002; Dathe 1997). Excess
        charge does not improve membrane disruption past the plateau and
        correlates with increased haemolytic risk.

        The plateau is NOT at a fixed charge. Measured across 1,678
        E. coli sequences, the share of potent peptides by charge within
        length strata plateaus at +5..+6 for 10-17mers but only at +7..+8
        for 24-30mers, where 94% are potent. That tracks
        ``charge = 0.35 x length``, so a fixed +5 peak penalises a long
        peptide for precisely the charge that makes it work.

    length:
        Peptide length. When given, the bell peaks at ``0.35 x length``
        clamped to [3, 9]. Omitted, the peak stays at +5 so existing
        callers are unaffected.

        ``"linear"``: ``min(1, max(0, charge / 10))``.
        Monotonically rewards higher charge; useful for models optimising
        LPS/LTA electrostatic displacement only.
    """
    if mode == "bell":
        # Rounded because net charge is an integer: a fractional peak shifts
        # every score slightly without meaning anything. Rounding also keeps
        # short peptides bit-identical to the fixed +5 this replaced.
        peak = 5.0 if length is None else float(
            min(9, max(3, int(round(0.35 * length)))))
        return max(0.0, 1.0 - abs(charge - peak) / _BELL_WIDTH)
    elif mode == "linear":
        return min(1.0, max(0.0, charge / 10.0))
    else:
        raise ValueError(
            f"normalize_charge: unknown mode {mode!r}. Use 'bell' or 'linear'."
        )


@functools.lru_cache(maxsize=8192)
def compute_hydrophobicity(sequence: str) -> float:
    """Mean Kyte-Doolittle hydrophobicity normalised to [0, 1]."""
    seq = sequence.upper()
    if not seq:
        return 0.0
    return sum(_KD_NORM.get(aa, 0.5) for aa in seq) / len(seq)


@functools.lru_cache(maxsize=8192)
def compute_amphipathicity(sequence: str) -> float:
    """Helical amphipathicity via the Eisenberg hydrophobic moment.

    Until 2026-09-30 this computed the moment from the Kyte-Doolittle
    table while documenting itself as Eisenberg, which put tryptophan and
    tyrosine on the polar face.

    Approximates the mean hydrophobic moment per residue using the standard
    100° helical wheel angle between successive residues.
    """
    seq = sequence.upper()
    n = len(seq)
    if n < 4:
        return 0.0
    angle_step = math.radians(100.0)
    hx, hy = 0.0, 0.0
    for i, aa in enumerate(seq):
        h = _EISENBERG.get(aa, 0.0)
        theta = i * angle_step
        hx += h * math.cos(theta)
        hy += h * math.sin(theta)
    moment = math.sqrt(hx ** 2 + hy ** 2) / n
    # Perfect alignment on the wheel cannot exceed the scale's extreme.
    return min(moment / _EISENBERG_MAX, 1.0)


def compute_membrane_insertion_score(sequence: str) -> float:
    """Estimate membrane-insertion affinity using Wimley–White ΔG.

    Returns a score in [0, 1] where 1 = strong insertion propensity
    (very negative mean ΔG) and 0 = membrane aversion.

    The reference range is calibrated for PE/PG-rich bacterial inner
    membranes (gram-negative ~75% PE / 20% PG; gram-positive ~60% PG)
    which are more polar than POPC, making interfacial insertion ~0.5
    kcal/mol more favourable.  The mapping is therefore shifted relative
    to the pure-POPC WW scale: mean_dg = -2.0 → score 1.0 (strong
    bacterial insertion); mean_dg = +1.5 → score 0.0 (aversion).
    """
    seq = sequence.upper()
    if not seq:
        return 0.0
    mean_dg = sum(_WW.get(aa, 0.0) for aa in seq) / len(seq)
    # Bacterial PE/PG membrane range: [-2.0, +1.5] kcal/mol per residue.
    # Shifted 0.5 kcal/mol more negative than POPC to reward sequences
    # with affinity for PG-rich bacterial bilayers.
    return max(0.0, min(1.0, (-mean_dg + 1.5) / 3.5))


def compute_membrane_context_risk(
    sequence: str,
    *,
    net_charge: float | None = None,
    hydrophobicity: float | None = None,
    amphipathicity: float | None = None,
    membrane_insertion_score: float | None = None,
    helix_propensity: float | None = None,
) -> float:
    """Contextual host-membrane risk proxy in [0, 1].

    AMPs need cationic charge and amphipathic membrane insertion, but the same
    features become selectivity risks when they co-occur too strongly in long,
    hydrophobic helices.  This proxy is deliberately conservative: it is near
    zero for compact +3 to +6 AMPs and rises for the high-charge,
    high-hydrophobic-moment profiles that tend to chase potency at the expense
    of RBC safety.
    """
    seq = sequence.upper().strip()
    n = len(seq)
    if n == 0:
        return 0.0
    charge = compute_net_charge(seq) if net_charge is None else net_charge
    hyd = compute_hydrophobicity(seq) if hydrophobicity is None else hydrophobicity
    amph = compute_amphipathicity(seq) if amphipathicity is None else amphipathicity
    memb = (
        compute_membrane_insertion_score(seq)
        if membrane_insertion_score is None else membrane_insertion_score
    )
    helix = compute_helix_propensity(seq) if helix_propensity is None else helix_propensity

    charge_excess = max(0.0, min(1.0, (charge - 5.0) / 5.0))
    hydrophobic_excess = max(0.0, min(1.0, (hyd - 0.55) / 0.35))
    amphipathic_excess = max(0.0, min(1.0, (amph - 0.45) / 0.35))
    insertion_excess = max(0.0, min(1.0, (memb - 0.40) / 0.35))
    length_excess = max(0.0, min(1.0, (n - 22.0) / 18.0))
    helix_excess = max(0.0, min(1.0, (helix - 0.70) / 0.30))

    return max(0.0, min(1.0,
        charge_excess * 0.32
        + hydrophobic_excess * 0.16
        + amphipathic_excess * 0.16
        + insertion_excess * 0.16
        + length_excess * 0.12
        + helix_excess * 0.08
    ))


def compute_stability_proxy(sequence: str) -> float:
    """Heuristic protease-stability score.

    Counts F and Y only. Trypsin sites (K/R) are deliberately excluded -- see
    _CLEAVAGE_SITES -- and W is omitted as well, although chymotrypsin cleaves
    F, Y and W alike. Adding W would change every composite score, so it is
    left as a deliberate open question rather than silently corrected.
    Fewer cleavage sites → higher stability score.
    Score is 1.0 when there are no cleavage sites, and decreases linearly.
    """
    seq = sequence.upper()
    if not seq:
        return 0.0
    n_cleavage = sum(1 for aa in seq if aa in _CLEAVAGE_SITES)
    # Allow up to 1 cleavage site per 10 residues before penalising strongly.
    allowed = max(1, len(seq) // 10)
    penalty = max(0.0, n_cleavage - allowed) / len(seq)
    return max(0.0, 1.0 - penalty * 5.0)


@functools.lru_cache(maxsize=8192)
def _compute_helix_propensity_chou_fasman(sequence: str) -> float:
    """Rough α-helix propensity using Chou-Fasman parameters (normalised).

    Fallback path when no Esm3HelixPredictor cache entry exists for this
    sequence. Cheap (just averaged per-residue parameters from 1978) but
    coarse — averaged scalar over residues, not an actual structure
    prediction. Use :func:`compute_helix_propensity` as the public entry
    point; it consults the singleton predictor's cache before falling back
    to this heuristic.
    """
    # Chou-Fasman P_alpha values (×100 for integer precision in literature)
    _CF_ALPHA: dict[str, float] = {
        "E": 1.51, "A": 1.42, "L": 1.21, "M": 1.45, "Q": 1.11, "K": 1.16,
        "R": 0.98, "H": 1.00, "V": 1.06, "I": 1.08, "Y": 0.69, "C": 0.70,
        "W": 1.08, "F": 1.13, "T": 0.83, "D": 1.01, "N": 0.67, "S": 0.77,
        "G": 0.57, "P": 0.57,
    }
    seq = sequence.upper()
    if not seq:
        return 0.0
    mean_p = sum(_CF_ALPHA.get(aa, 1.0) for aa in seq) / len(seq)
    # Scale: P_alpha = 1.0 → helix-indifferent; > 1.0 → helix-former
    return min(1.0, max(0.0, (mean_p - 0.5) / 1.0))


def compute_helix_propensity(sequence: str) -> float:
    """α-helix propensity in [0, 1].

    Lookup priority (mirrors :class:`HemolysisPredictor` pattern):

      1. Esm3HelixPredictor cache (real per-residue DSSP-style secondary-
         structure prediction from ESM3-open via the Biohub Forge API).
         Populated by :meth:`Esm3HelixPredictor.predict_batch` on the
         top-N candidates each round.
      2. Chou-Fasman heuristic (averaged 1978-era P_alpha parameters).

    Falling back to Chou-Fasman keeps fast-scoring cheap when no Forge
    key is set or for sequences outside the pre-filled batch; the lift
    only kicks in for top-N candidates that actually matter for ranking.
    """
    seq_u = sequence.upper().strip()
    p = _helix_predictor
    if p is not None and p.is_available and seq_u in p._cache:
        return float(p._cache[seq_u])
    return _compute_helix_propensity_chou_fasman(sequence)


def _forge_helix_fraction(
    sequence: str, token: str, url: str, timeout: int
) -> "tuple[float | None, str]":
    """Single-sequence Forge call — returns ``(helix_fraction, error)``.

    Calls ESM3-open ``generate(track="secondary_structure")`` and counts the
    'H'/'G'/'I' characters in the returned DSSP-style string.

    Returns ``(None, reason)`` on any failure rather than the Chou-Fasman
    value. Returning the heuristic here was actively harmful: the caller
    cached it, :func:`compute_helix_propensity` then served it as though it
    came from ESM3, and the run manifest reported the helix signal as
    available because a token was merely *present*. A rejected credential
    therefore produced a full run scored entirely on the 1978 heuristic while
    every status check said otherwise — the exact v14_full failure the W8
    strict_signals guard was built to stop, slipping straight past that guard.
    """
    import logging
    log = logging.getLogger(__name__)
    try:
        from esm.sdk.forge import ESM3ForgeInferenceClient
        from esm.sdk.api import ESMProtein, GenerationConfig
        client = ESM3ForgeInferenceClient(
            model="esm3-open-2024-03", url=url, token=token, request_timeout=timeout,
        )
        prot = ESMProtein(sequence=sequence.upper().strip())
        out = client.generate(
            prot,
            GenerationConfig(
                track="secondary_structure", num_steps=10, temperature=0.0,
            ),
        )
        ss = getattr(out, "secondary_structure", None)
        if not ss or not isinstance(ss, str):
            # ESMProteinError arrives as a normal return value, not a raise,
            # so its message is the only evidence of an auth/quota problem.
            reason = str(
                getattr(out, "error_msg", None) or type(out).__name__
            )[:200]
            return None, reason
        n = len(ss)
        if n == 0:
            return None, "empty secondary_structure"
        # DSSP convention: 'H' = α-helix, 'G' = 3_10-helix, 'I' = π-helix.
        # We count all three as "helix" since they share the helical-bundle
        # topology that drives membrane insertion in AMPs.
        helix_residues = sum(1 for ch in ss if ch in ("H", "G", "I"))
        return float(helix_residues) / float(n), ""
    except Exception as exc:
        log.debug("Forge helix fetch failed for %s: %s", sequence[:10], exc)
        return None, f"{type(exc).__name__}: {exc}"[:200]


class Esm3HelixPredictor:
    """Real α-helix predictor for top-N candidates via ESM3-open over Forge.

    Designed as a batch pre-filler: each pipeline round calls
    :meth:`predict_batch` once on the deep-eval candidates; subsequent
    per-sequence :func:`compute_helix_propensity` calls then hit the cache
    directly. Parallelises Forge HTTP calls with a thread pool because the
    bottleneck is per-call latency (~1.8 s), not bandwidth.

    Available when ``ESM_API_KEY`` is set in the environment. Otherwise
    :attr:`is_available` is False and :meth:`predict_batch` is a no-op,
    so callers can safely invoke it unconditionally.
    """

    def __init__(
        self,
        url: str | None = None,
        max_workers: int = 3,
        request_timeout: int = 60,
        cache_path: str | None = None,
    ) -> None:
        import os
        # Endpoint is overridable: "https://biohub.ai" and
        # "https://forge.evolutionaryscale.ai" both speak the Forge API and a
        # token is usually valid for only one of them. Set AMP_ESM_FORGE_URL
        # to point at whichever issued the key.
        self._url = url or os.environ.get(
            "AMP_ESM_FORGE_URL", "https://biohub.ai"
        ).strip()
        # Forge esm3-open is capped at 50 req/min; with ~1.8 s/req, 3 workers
        # land around 100 req/min — the SDK's retry/backoff absorbs the overage
        # without thrashing. Higher concurrency just inflates wall time.
        self._max_workers = max_workers
        self._request_timeout = request_timeout
        self._token = os.environ.get("ESM_API_KEY", "").strip()
        self._cache: dict[str, float] = {}
        # Persistent cache. ESM3 is generative: the helix fraction comes from a
        # sampled structure, so the same peptide comes back with a different
        # number on a later call. Two runs of identical code with the same seed
        # disagreed on 23% of the sequences their top-100s had in common, by up
        # to 0.7333, which moved the composite by 0.121 and reordered the
        # ranking. Keeping the first answer on disk makes repeated runs agree.
        self._cache_path = (
            cache_path
            or os.environ.get("AMP_HELIX_CACHE", "").strip()
            or "data/cache/esm3_helix.json"
        )
        self._load_cache()
        # None until a batch has been attempted; then True only if the
        # endpoint actually returned a structure for at least one sequence.
        self._working: bool | None = None
        self._last_error: str = ""

    def _load_cache(self) -> int:
        """Read stored predictions. A missing or unreadable file is not fatal."""
        import json
        import logging
        log = logging.getLogger(__name__)
        try:
            with open(self._cache_path, encoding="utf-8") as fh:
                data = json.load(fh)
        except FileNotFoundError:
            return 0
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "Esm3HelixPredictor: could not read the cache at %s (%s); "
                "continuing without it.", self._cache_path, exc,
            )
            return 0
        n = 0
        for key, value in (data or {}).items():
            try:
                self._cache[str(key).upper().strip()] = float(value)
                n += 1
            except (TypeError, ValueError):
                continue
        if n:
            log.info(
                "Esm3HelixPredictor: %d stored prediction(s) loaded from %s",
                n, self._cache_path,
            )
        return n

    def _save_cache(self) -> None:
        """Write the cache atomically, so an interrupted run cannot corrupt it."""
        import json
        import logging
        import os
        import tempfile
        log = logging.getLogger(__name__)
        try:
            folder = os.path.dirname(self._cache_path)
            if folder:
                os.makedirs(folder, exist_ok=True)
            handle, tmp = tempfile.mkstemp(dir=folder or ".", suffix=".tmp")
            with os.fdopen(handle, "w", encoding="utf-8") as fh:
                json.dump(self._cache, fh, sort_keys=True)
            os.replace(tmp, self._cache_path)
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "Esm3HelixPredictor: could not write the cache at %s (%s); "
                "this run stays reproducible only in memory.",
                self._cache_path, exc,
            )

    @property
    def is_available(self) -> bool:
        """True when a Forge token is configured.

        Token presence only — it says nothing about whether the endpoint
        accepts that token. Use :attr:`is_working` before trusting any helix
        value as a real prediction.
        """
        return bool(self._token)

    @property
    def is_working(self) -> bool | None:
        """True/False once a batch has run; None while untried."""
        return self._working

    @property
    def last_error(self) -> str:
        """Most recent Forge failure message (empty when healthy)."""
        return self._last_error

    @property
    def endpoint(self) -> str:
        return self._url

    @property
    def cache_size(self) -> int:
        return len(self._cache)

    def predict_batch(self, sequences: Sequence[str]) -> dict[str, float]:
        """Pre-fill the cache for many sequences via parallel Forge calls.

        Returns the {upper-case sequence → helix fraction} entries added
        in this call. Missing sequences (cache hits, empty input, or a
        no-token install) yield an empty dict.
        """
        if not self.is_available or not sequences:
            return {}
        import concurrent.futures
        import logging
        log = logging.getLogger(__name__)

        # Skip sequences already cached.
        todo: list[str] = []
        seen: set[str] = set()
        for s in sequences:
            s_u = s.upper().strip()
            if s_u and s_u not in self._cache and s_u not in seen:
                seen.add(s_u)
                todo.append(s_u)
        if not todo:
            return {}

        results: dict[str, float] = {}
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self._max_workers,
        ) as pool:
            futures = {
                pool.submit(
                    _forge_helix_fraction,
                    seq, self._token, self._url, self._request_timeout,
                ): seq
                for seq in todo
            }
            for fut in concurrent.futures.as_completed(futures):
                seq = futures[fut]
                try:
                    value, reason = fut.result()
                except Exception as exc:
                    value, reason = None, f"{type(exc).__name__}: {exc}"[:200]
                if value is None:
                    if reason and not self._last_error:
                        self._last_error = reason
                    continue
                results[seq] = value
        # Only real predictions enter the cache. A heuristic value cached here
        # would be indistinguishable from an ESM3 one downstream.
        self._cache.update(results)
        if results:
            self._save_cache()
        self._working = bool(results)
        if results:
            self._last_error = ""
            log.info(
                "Esm3HelixPredictor: scored %d / %d sequences via %s",
                len(results), len(todo), self._url,
            )
        else:
            log.error(
                "Esm3HelixPredictor: 0 / %d sequences scored at %s (%s). "
                "helix_propensity is the 1978 Chou-Fasman heuristic for this "
                "run. Check ESM_API_KEY, or point AMP_ESM_FORGE_URL at the "
                "endpoint that issued the key.",
                len(todo), self._url, self._last_error or "no reason reported",
            )
        return results


# Module-level singleton — shared with AMPScorer just like HemolysisPredictor.
# Constructed lazily on first access so the env var lookup happens once.
_helix_predictor: "Esm3HelixPredictor | None" = None


def get_helix_predictor() -> Esm3HelixPredictor:
    """Return (or lazily create) the module-level Esm3HelixPredictor."""
    global _helix_predictor
    if _helix_predictor is None:
        _helix_predictor = Esm3HelixPredictor()
    return _helix_predictor


def _repeat_kmer_penalty(sequence: str, kmer_len: int = 6) -> float:
    """Return a graded penalty for recurrent k-mers in ``sequence``.

    The compliance gate rejects any repeated 6-mer outright.  The scorer uses
    this softer count-based version so borderline repeat-prone candidates are
    deprioritised during fast screening before the final hard filter runs.
    """
    seq = sequence.upper()
    if len(seq) < kmer_len * 2:
        return 0.0
    seen: set[str] = set()
    repeats = 0
    for i in range(len(seq) - kmer_len + 1):
        kmer = seq[i:i + kmer_len]
        if kmer in seen:
            repeats += 1
        else:
            seen.add(kmer)
    return min(1.0, repeats / max(1, len(seq) - kmer_len + 1))


def compute_complexity_score(sequence: str) -> float:
    """Sequence complexity in [0, 1] = entropy minus repeat/artifact penalties.

    Real AMPs in the training set have ≥ 8 distinct residues (entropy_norm ≈ 0.7).
    Synthetic outputs from base PepMLM-650M often collapse to repeats like
    ``PGPGPGPG`` or ``GLMQFIKR...GLMQFIKR``.  This score rewards real-AMP-like
    distributions and penalises both long single-residue runs and recurrent
    k-mers that the final compliance filter would reject.
    """
    seq = sequence.upper()
    n = len(seq)
    if n < 2:
        return 0.0

    from collections import Counter as _C
    counts = _C(seq)
    # Shannon entropy normalised against log2(20). Real AMPs hit ~0.65–0.85.
    h = -sum((c / n) * math.log2(c / n) for c in counts.values())
    entropy_norm = h / math.log2(20)  # 1.0 = uniform across 20 AAs

    # Run-length penalty: count any residue appearing ≥ 4 times consecutively
    run_penalty = 0.0
    if n >= 4:
        run_len = 1
        for i in range(1, n):
            if seq[i] == seq[i - 1]:
                run_len += 1
                if run_len >= 4:
                    # Each extra residue in a long run adds a unit of penalty
                    run_penalty += 1.0
            else:
                run_len = 1
        run_penalty = min(1.0, run_penalty / n)

    try:
        from .compliance import MAX_REPEATED_KMER_LEN, has_low_complexity_repeat
        repeat_k = MAX_REPEATED_KMER_LEN + 1
        repeat_penalty = (
            _repeat_kmer_penalty(seq, repeat_k)
            if has_low_complexity_repeat(seq, repeat_k)
            else 0.0
        )
    except Exception:
        repeat_penalty = _repeat_kmer_penalty(seq, 6)

    return max(0.0, min(1.0, entropy_norm - 0.6 * run_penalty - 0.8 * repeat_penalty))


# ---------------------------------------------------------------------------
# Composite AMP scorer
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class AMPScoreResult:
    """All per-residue and composite scores for a candidate peptide."""
    sequence: str
    net_charge: float
    hydrophobicity: float
    amphipathicity: float
    membrane_insertion_score: float
    stability_proxy: float
    helix_propensity: float
    hemolysis_risk_proxy: float
    selectivity_score: float
    composite_amp_score: float
    membrane_context_risk: float = 0.0


class AMPScorer:
    """Compute AMP-specific heuristic scores for a batch of sequences.

    Parameters
    ----------
    activity_weight, selectivity_weight, stability_weight, structural_weight:
        Linear weights for the composite AMP score.  Values are normalised
        internally so they sum to 1.
    min_length, max_length:
        Sequences outside this range receive a zero composite score.
    min_charge:
        Sequences with net_charge < min_charge receive a zero composite score.
    """

    def __init__(
        self,
        activity_weight: float = 2.0,    # ↑ membrane disruption is primary goal
        selectivity_weight: float = 1.0,  # anti-hemolytic safety maintained
        stability_weight: float = 0.8,
        structural_weight: float = 1.0,   # ↑ α-helix essential for pore-forming
        complexity_weight: float = 0.6,   # rewards sequence diversity (entropy + no-runs)
        min_length: int = 8,
        max_length: int = 40,
        min_charge: float = 1.0,
        activity_membrane_weight: float = 0.35,
        activity_amphipathicity_weight: float = 0.35,
        activity_charge_weight: float = 0.30,
    ) -> None:
        total = (activity_weight + selectivity_weight + stability_weight
                 + structural_weight + complexity_weight)
        self._w_act = activity_weight / total
        self._w_sel = selectivity_weight / total
        self._w_stab = stability_weight / total
        self._w_struct = structural_weight / total
        self._w_complex = complexity_weight / total
        self._min_len = min_length
        self._max_len = max_length
        self._min_charge = min_charge
        # Activity sub-weights — shared with AMPEvaluator so both screening
        # stages rank sequences with the same activity signal.
        self._act_w_memb   = activity_membrane_weight
        self._act_w_amph   = activity_amphipathicity_weight
        self._act_w_charge = activity_charge_weight

    def _hemolysis_risk(self, sequence: str, net_charge: float, hydrophobicity: float) -> float:
        """Hemolysis risk in [0, 1] via calibrated LR model or heuristic fallback.

        Delegates to the module-level HemolysisPredictor singleton which
        loads ``data/external/hemolysis_lr_model.json`` when available.
        Reference: Fjell et al., 2012; Wang et al., 2022.
        """
        return get_hemolysis_predictor().predict(sequence)

    def score(self, sequence: str) -> AMPScoreResult:
        """Score a single sequence."""
        seq = sequence.upper().strip()
        n = len(seq)

        # Length / charge gate
        if n < self._min_len or n > self._max_len:
            zero = AMPScoreResult(
                sequence=seq, net_charge=0.0, hydrophobicity=0.0,
                amphipathicity=0.0, membrane_insertion_score=0.0,
                stability_proxy=0.0, helix_propensity=0.0,
                hemolysis_risk_proxy=1.0, selectivity_score=0.0,
                composite_amp_score=0.0,
                membrane_context_risk=1.0,
            )
            return zero

        charge = compute_net_charge(seq)
        if charge < self._min_charge:
            zero = AMPScoreResult(
                sequence=seq, net_charge=charge, hydrophobicity=0.0,
                amphipathicity=0.0, membrane_insertion_score=0.0,
                stability_proxy=0.0, helix_propensity=0.0,
                hemolysis_risk_proxy=1.0, selectivity_score=0.0,
                composite_amp_score=0.0,
                membrane_context_risk=1.0,
            )
            return zero

        hyd   = compute_hydrophobicity(seq)
        amph  = compute_amphipathicity(seq)
        memb  = compute_membrane_insertion_score(seq)
        stab  = compute_stability_proxy(seq)
        helix = compute_helix_propensity(seq)
        hem   = self._hemolysis_risk(seq, charge, hyd)
        cplx  = compute_complexity_score(seq)
        membrane_risk = compute_membrane_context_risk(
            seq,
            net_charge=charge,
            hydrophobicity=hyd,
            amphipathicity=amph,
            membrane_insertion_score=memb,
            helix_propensity=helix,
        )
        selectivity = max(0.0, 1.0 - hem)

        # Activity: membrane insertion × amphipathicity × charge contribution.
        # The charge bell peaks at 0.35 × length rather than a fixed +5: the
        # plateau measured across 1,678 E. coli sequences moves with peptide
        # size, from +5..+6 for 10-17mers to +7..+8 for 24-30mers.
        # Sub-weights are instance-configurable so AMPEvaluator can use the
        # same values and both scoring stages produce a consistent ranking.
        charge_norm = normalize_charge(charge, mode="bell",
                                       length=len(sequence))
        activity = (memb * self._act_w_memb
                    + amph * self._act_w_amph
                    + charge_norm * self._act_w_charge)

        composite = (
            self._w_act    * activity
            + self._w_sel  * selectivity
            + self._w_stab * stab
            + self._w_struct * helix
            + self._w_complex * cplx
            - 0.10 * membrane_risk
        )
        composite = max(0.0, min(1.0, composite))

        return AMPScoreResult(
            sequence=seq,
            net_charge=charge,
            hydrophobicity=hyd,
            amphipathicity=amph,
            membrane_insertion_score=memb,
            stability_proxy=stab,
            helix_propensity=helix,
            hemolysis_risk_proxy=hem,
            selectivity_score=selectivity,
            composite_amp_score=composite,
            membrane_context_risk=membrane_risk,
        )

    def score_batch(self, sequences: Sequence[str]) -> list[AMPScoreResult]:
        """Score a list of sequences using the cached HemoPI2 / LR predictors.

        HemoPI2 cache pre-fill is the pipeline's responsibility — see
        :meth:`AMPChallengerPipeline.generate_library` which calls
        :meth:`HemolysisPredictor.predict_batch` once on the deep-eval batch
        (typically a few hundred sequences) so every `.predict()` here is a
        dict look-up. Earlier we eagerly pre-filled inside `score_batch`,
        but MCTS / SwarmRefiner / Epistasis each call this with small
        batches of ~8 mutants per iteration, and the eager call spawned a
        fresh ~3 s HemoPI2 subprocess per batch — turning a 5-minute run
        into 90 minutes. Per-mutant misses now fall back to the calibrated
        LR / heuristic, which is the correct choice for exploration
        scoring; only the final deep-eval pass needs HemoPI2 quality.
        """
        return [self.score(seq) for seq in sequences]


# ---------------------------------------------------------------------------
# Calibrated hemolysis predictor
# ---------------------------------------------------------------------------

# Hydrophobic residues used by the hemolysis LR feature extractor.
_HEMO_HYDROPHOBIC: frozenset[str] = frozenset("ILVFMWAP")
_HEMO_AROMATIC: frozenset[str] = frozenset("FYW")
_HEMO_BASIC: frozenset[str] = frozenset("KRH")
_HEMO_ACIDIC: frozenset[str] = frozenset("DE")
_HEMO_POLAR: frozenset[str] = frozenset("STNQCY")

# Names of the features used by the hemolysis models. Single source of
# truth so the training script and the inference predictor cannot drift.
HEMOLYSIS_FEATURE_NAMES: list[str] = [
    "net_charge", "hydrophobic_fraction", "amphipathicity",
    "length_norm", "trp_fraction", "phe_fraction", "leu_fraction",
    "aromatic_fraction", "basic_fraction", "acidic_fraction",
    "polar_fraction", "gly_fraction", "pro_fraction", "lys_arg_balance",
    "aa_diversity_fraction", "adjacent_repeat_fraction",
]


def extract_hemolysis_features(sequence: str) -> list[float]:
    """Compute the features consumed by the hemolysis models.

    Used by both ``HemolysisPredictor`` (inference) and
    ``scripts/train_hemolysis_model.py`` (training) to guarantee the two
    paths see the same feature vector. Feature order matches
    ``HEMOLYSIS_FEATURE_NAMES``.
    """
    seq = sequence.upper()
    n = max(len(seq), 1)
    hydrophobic_fraction = sum(1 for aa in seq if aa in _HEMO_HYDROPHOBIC) / n
    basic_count = sum(1 for aa in seq if aa in _HEMO_BASIC)
    acidic_count = sum(1 for aa in seq if aa in _HEMO_ACIDIC)
    kr_count = seq.count("K") + seq.count("R")
    repeat_fraction = (
        sum(1 for i in range(1, len(seq)) if seq[i] == seq[i - 1])
        / max(len(seq) - 1, 1)
    )
    return [
        compute_net_charge(seq),
        hydrophobic_fraction,
        compute_amphipathicity(seq),
        min(n / 50.0, 1.0),
        seq.count("W") / n,
        seq.count("F") / n,
        seq.count("L") / n,
        sum(1 for aa in seq if aa in _HEMO_AROMATIC) / n,
        basic_count / n,
        acidic_count / n,
        sum(1 for aa in seq if aa in _HEMO_POLAR) / n,
        seq.count("G") / n,
        seq.count("P") / n,
        seq.count("K") / max(kr_count, 1),
        len(set(seq)) / 20.0,
        repeat_fraction,
    ]


# Natural-AMP composition envelope — the same guardrails the competition
# submission selector enforces (build_competition_submission.py). Kept here so
# the in-loop generator-quality signal can share one definition of "artifact".
REALISM_MAX_HIS_FRACTION: float = 0.15
REALISM_MAX_NET_CHARGE: int = 9
# The former REALISM_MIN_COMPLEXITY = 0.45 (distinct residues / length) was
# a length filter: on the 17-letter effective alphabet it rejected 49.7 % of
# the project's own validated AMPs, including every one of 45 aa or longer.
# Complexity now goes through the per-length calibrated ComplexityGate.


def composition_realism_factor(
    sequence: str,
    *,
    max_his: float = REALISM_MAX_HIS_FRACTION,
    max_charge: int = REALISM_MAX_NET_CHARGE,
    complexity_gate: "ComplexityGate | None" = None,
) -> float:
    """Return 1.0 if a sequence is in the natural-AMP regime, else 0.0.

    Flags the oracle-exploiting Goodhart artifacts — His-stacking, hyper-
    cationicity, and low-complexity repeats — that inflate in-silico scores
    without biophysical basis. Multiplying a generator's per-sequence score by
    this factor makes the adaptive reweighting signal robust: a generator that
    games the oracle via these artifacts has those candidates zeroed, so the
    reasoner stops up-weighting it (turning the loop from a Goodhart amplifier
    into a generalisation selector).
    """
    n = max(len(sequence), 1)
    his = sequence.count("H") / n
    charge = (sequence.count("K") + sequence.count("R")) - (
        sequence.count("D") + sequence.count("E")
    )
    gate = complexity_gate if complexity_gate is not None else get_complexity_gate()
    ok = his <= max_his and charge <= max_charge and gate.passes(sequence)
    return 1.0 if ok else 0.0


def _run_hemopi2_batch(
    sequences: list[str],
    model: int = 4,
    job: int = 1,
) -> dict[str, float]:
    """Call ``hemopi2_classification`` once on a batch and return {seq → ESM Score}.

    HemoPI2.0 (Rathore et al., Raghava group) trains an ESM2-t6 classifier +
    MERCI motif model on a curated hemolytic-peptide database. Model 4
    (Hybrid2 = ESM + MERCI) is the highest-quality default. We use the
    continuous ``ESM Score`` column as our hemolysis probability — it's
    monotonic in risk and has finer granularity than the binary prediction
    label.

    Returns an empty dict on any failure so the caller can fall back to the
    LR / heuristic without raising.
    """
    import logging
    import shutil
    import subprocess
    import tempfile

    _log = logging.getLogger(__name__)
    if not sequences:
        return {}
    if shutil.which("hemopi2_classification") is None:
        _log.debug("hemopi2_classification not on PATH — skipping batch.")
        return {}

    seen: set[str] = set()
    uniq: list[str] = []
    for s in sequences:
        s2 = s.upper().strip()
        if s2 and s2 not in seen:
            seen.add(s2)
            uniq.append(s2)
    if not uniq:
        return {}

    tmp = Path(tempfile.mkdtemp(prefix="hemopi2_"))
    try:
        # HemoPI2 internally builds the output path as f"{wd}/{result_filename}"
        # so we MUST pass -o as a bare filename (not an absolute path) and let
        # HemoPI2 join it to the working dir we supply via -wd.
        fa_name = "in.fa"
        out_name = "out.csv"
        fa = tmp / fa_name
        out = tmp / out_name
        with open(fa, "w", encoding="utf-8") as fh:
            for i, seq in enumerate(uniq):
                fh.write(f">s{i}\n{seq}\n")
        cmd = [
            "hemopi2_classification",
            "-i", str(fa), "-o", out_name,
            "-j", str(job), "-m", str(model),
            "-wd", str(tmp),
        ]
        # Force CPU for the hemopi2 subprocess. Its model load tries CUDA by
        # default, but the parent pipeline already has ESM3 / ESMC / APEX /
        # CPL-Diff resident on the GPU, so the subprocess OOMs at model
        # init. ESM2-t6 (the model hemopi2 uses) is tiny — CPU inference
        # for ~200 sequences takes ~20 s, dominated by model load. Override
        # with HEMOPI2_CUDA_VISIBLE_DEVICES=0 if you want to keep GPU.
        import os as _os
        sub_env = _os.environ.copy()
        sub_env["CUDA_VISIBLE_DEVICES"] = _os.environ.get(
            "HEMOPI2_CUDA_VISIBLE_DEVICES", ""
        )
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300,
            cwd=str(tmp), env=sub_env,
        )
        if proc.returncode != 0:
            _log.warning("hemopi2 returned %d: %s",
                         proc.returncode, (proc.stderr or "")[:200])
            return {}
        if not out.exists():
            _log.warning("hemopi2 produced no output file at %s", out)
            return {}
        # Parse CSV — header: SeqID, Sequence, ESM Score, MERCI Score, Hybrid Score, Prediction
        import csv as _csv
        results: dict[str, float] = {}
        with open(out, "r", encoding="utf-8") as fh:
            reader = _csv.DictReader(fh)
            for row in reader:
                seq = (row.get("Sequence") or "").strip().upper()
                if not seq:
                    continue
                try:
                    # Use ESM Score (continuous probability of hemolytic).
                    # MERCI / Hybrid often missing (-1, 0) when no motif hit,
                    # so ESM alone is the most consistently informative column.
                    p = float(row.get("ESM Score", "nan"))
                except ValueError:
                    continue
                if 0.0 <= p <= 1.0:
                    results[seq] = p
        _log.info("hemopi2: scored %d / %d sequences", len(results), len(uniq))
        return results
    except subprocess.TimeoutExpired:
        _log.warning("hemopi2 timed out on %d sequences", len(uniq))
        return {}
    except Exception as exc:
        _log.warning("hemopi2 batch failed: %s", exc)
        return {}
    finally:
        # Best-effort cleanup of working dir.
        try:
            shutil.rmtree(tmp, ignore_errors=True)
        except Exception:
            pass


class HemolysisPredictor:
    """Hemolysis probability predictor with three backends, in priority order.

    1. **HemoPI2** (Raghava lab, ESM2-t6 + MERCI hybrid). Called in batch via
       :func:`_run_hemopi2_batch` and cached per-sequence. Highest quality;
       requires ``pip install hemopi2``. Use :meth:`predict_batch` to pre-fill
       the cache before scoring loops to amortise the ~3 s model-load cost.
    2. **Calibrated logistic regression** trained by
       ``scripts/train_hemolysis_model.py`` and serialised to
       ``data/external/hemolysis_lr_model.json``.
    3. **Heuristic fallback** (Fjell et al. 2012; Wang et al. 2022).

    Model JSON format (produced by ``scripts/train_hemolysis_model.py``)::

        {
            "coef":      [c1, c2, ..., c7],   # legacy risk model alias
            "intercept": float,                # legacy risk model alias
            "risk_coef": [c1, c2, ..., cN],
            "risk_intercept": float,
            "hc50_coef": [c1, c2, ..., cN],
            "hc50_intercept": float,
            "features":  ["net_charge", "hydrophobic_fraction",
                          "amphipathicity", "length_norm",
                          "trp_fraction", "phe_fraction", "leu_fraction"]
        }

    Feature computation uses the same functions as ``AMPScorer`` so that
    results are consistent throughout the pipeline.
    """

    # Expected feature names (order matches model coefficients).
    _FEATURE_NAMES: tuple[str, ...] = tuple(HEMOLYSIS_FEATURE_NAMES)

    def __init__(
        self,
        model_path: str | None = None,
        use_hemopi2: bool = True,
    ) -> None:
        """Load LR model from ``model_path`` and prepare the HemoPI2 cache.

        Parameters
        ----------
        model_path:
            Explicit path to ``hemolysis_lr_model.json``.  When ``None``, the
            constructor searches relative to the package's ``data/external/``
            directory (two levels above ``amp_scoring.py``).
        use_hemopi2:
            When True (default), :meth:`predict` and :meth:`predict_batch`
            prefer cached HemoPI2 results when available. Pass False to
            force the LR / heuristic backend (useful for unit tests).
        """
        import json
        import logging

        self._coef: list[float] | None = None
        self._intercept: float = 0.0
        self._model_available: bool = False
        self._hc50_coef: list[float] | None = None
        self._hc50_intercept: float = 0.0
        self._hc50_log10_min: float = -1.0
        self._hc50_log10_max: float = 4.0
        self._hc50_model_available: bool = False
        self._model_path: str = model_path or ""
        self._model_metadata: dict[str, object] = {}
        # HemoPI2 cache: {upper-cased sequence → ESM probability}.
        # Filled by :meth:`predict_batch` on the top-N candidates each round.
        self._hemopi2_cache: dict[str, float] = {}
        # Honest availability: only claim HemoPI2 if the CLI is actually on
        # PATH. Without this check, the predictor advertises "pre-filled
        # cache (0 entries)" on every round while silently falling back to
        # the LR / heuristic — a misleading false success in the logs.
        import shutil as _shutil
        self._use_hemopi2: bool = bool(
            use_hemopi2 and _shutil.which("hemopi2_classification") is not None
        )
        if use_hemopi2 and not self._use_hemopi2:
            logging.getLogger(__name__).info(
                "HemoPI2: hemopi2_classification not on PATH — disabling. "
                "Install with `pip install hemopi2` to enable ESM2 hemolysis scoring."
            )

        if model_path is None:
            default = (
                Path(__file__).resolve().parents[2]
                / "data" / "external" / "hemolysis_lr_model.json"
            )
            model_path = str(default)
        self._model_path = model_path

        try:
            with open(model_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            self._coef = [
                float(c) for c in data.get("risk_coef", data["coef"])
            ]
            self._intercept = float(data.get("risk_intercept", data["intercept"]))
            self._model_available = True
            if "hc50_coef" in data and "hc50_intercept" in data:
                self._hc50_coef = [float(c) for c in data["hc50_coef"]]
                self._hc50_intercept = float(data["hc50_intercept"])
                self._hc50_log10_min = float(data.get("hc50_log10_min", -1.0))
                self._hc50_log10_max = float(data.get("hc50_log10_max", 4.0))
                self._hc50_model_available = True
            self._model_metadata = {
                str(k): v
                for k, v in data.items()
                if k not in {
                    "coef", "intercept", "risk_coef", "risk_intercept",
                    "hc50_coef", "hc50_intercept",
                }
            }
        except FileNotFoundError:
            logging.getLogger(__name__).warning(
                "HemolysisPredictor: model file not found at %s. "
                "Falling back to heuristic proxy. "
                "Run scripts/train_hemolysis_model.py to generate a "
                "calibrated model.",
                model_path,
            )
        except (KeyError, ValueError, json.JSONDecodeError) as exc:
            logging.getLogger(__name__).warning(
                "HemolysisPredictor: could not load model from %s (%s). "
                "Falling back to heuristic proxy.",
                model_path, exc,
            )

    @staticmethod
    def _extract_features(sequence: str) -> list[float]:
        """Compute the features used by the calibrated hemolysis models."""
        return extract_hemolysis_features(sequence)

    def predict(self, sequence: str) -> float:
        """Return hemolysis probability in [0, 1].

        Lookup priority: HemoPI2 cache → calibrated LR → heuristic fallback.
        Call :meth:`predict_batch` first to populate the HemoPI2 cache
        amortised across many sequences.
        """
        seq_u = sequence.upper().strip()
        if self._use_hemopi2:
            cached = self._hemopi2_cache.get(seq_u)
            if cached is not None:
                return float(cached)
        if self._model_available and self._coef is not None:
            feats = self._extract_features(sequence)
            logit = self._intercept + sum(c * f for c, f in zip(self._coef, feats))
            # sigmoid
            return 1.0 / (1.0 + math.exp(-max(-500.0, min(500.0, logit))))

        # Legacy heuristic fallback (Fjell et al. 2012; Wang et al. 2022)
        net_charge = compute_net_charge(seq_u)
        hyd = compute_hydrophobicity(seq_u)
        trp_fraction = seq_u.count("W") / max(len(seq_u), 1)
        hyd_penalty = max(0.0, hyd - 0.6) * 2.5
        charge_penalty = max(0.0, 2.0 - net_charge) * 0.2
        return min(1.0, hyd_penalty + charge_penalty + trp_fraction * 0.3)

    def predict_hc50_um(self, sequence: str) -> float:
        """Return predicted HC50 in µM.

        When the calibrated HC50 regressor is unavailable, this falls back to
        the legacy risk-derived pseudo-HC50 so callers can keep a continuous
        safety-window signal.  Use :attr:`has_hc50_model` to distinguish real
        calibrated predictions from that compatibility fallback.
        """
        if self._hc50_model_available and self._hc50_coef is not None:
            feats = self._extract_features(sequence)
            log10_hc50 = self._hc50_intercept + sum(
                c * f for c, f in zip(self._hc50_coef, feats)
            )
            log10_hc50 = min(
                max(log10_hc50, self._hc50_log10_min),
                self._hc50_log10_max,
            )
            return float(10.0 ** log10_hc50)

        risk = self.predict(sequence)
        floor = 1.0
        assay_ceiling = 128.0
        return floor + (assay_ceiling - floor) * (1.0 - min(max(risk, 0.0), 1.0))

    def predict_batch(
        self, sequences: Sequence[str], chunk_size: int = 2000
    ) -> dict[str, float]:
        """Pre-fill the HemoPI2 cache for many sequences via chunked subprocess calls.

        Returns the {upper-case sequence → ESM probability} mapping that was
        added to the cache. Sequences absent from the result (e.g. on a CLI
        failure) fall back to the LR / heuristic backend on subsequent
        :meth:`predict` calls.

        Large pools are split into ``chunk_size`` sub-batches: a single HemoPI2
        call on tens of thousands of sequences exhausts the subprocess timeout /
        memory and fails as a whole (returncode 1), silently dropping the entire
        ESM2 gate to the LR fallback. Chunking keeps each call bounded and lets a
        single bad chunk fail in isolation instead of taking down the run.
        """
        if not self._use_hemopi2 or not sequences:
            return {}
        # Skip sequences already cached.
        todo = [s.upper().strip() for s in sequences
                if s and s.upper().strip() not in self._hemopi2_cache]
        if not todo:
            return {}
        added: dict[str, float] = {}
        step = max(1, chunk_size)
        for start in range(0, len(todo), step):
            results = _run_hemopi2_batch(todo[start:start + step])
            if results:
                self._hemopi2_cache.update(results)
                added.update(results)
        return added

    @property
    def is_calibrated(self) -> bool:
        """True when a trained LR model is loaded (not using the heuristic)."""
        return self._model_available

    @property
    def has_hc50_model(self) -> bool:
        """True when a calibrated HC50 regressor is loaded."""
        return self._hc50_model_available

    @property
    def model_path(self) -> str:
        """Path used to load the calibrated hemolysis model."""
        return self._model_path

    @property
    def model_metadata(self) -> dict[str, object]:
        """Non-coefficient metadata from the loaded model JSON."""
        return dict(self._model_metadata)

    @property
    def hemopi2_cache_size(self) -> int:
        """Number of sequences currently cached via HemoPI2."""
        return len(self._hemopi2_cache)


# Module-level singleton — shared by AMPScorer and ResiDPOTracker.
# Keyed on the resolved model_path so callers that request a different
# calibrated model don't silently get the originally cached predictor.
_hemolysis_predictor: HemolysisPredictor | None = None
_hemolysis_predictor_path: str | None = None


def get_hemolysis_predictor(model_path: str | None = None) -> HemolysisPredictor:
    """Return (or lazily create) the module-level HemolysisPredictor singleton.

    Re-creates the predictor if ``model_path`` differs from the one used for
    the cached instance; otherwise returns the cache.
    """
    global _hemolysis_predictor, _hemolysis_predictor_path
    if _hemolysis_predictor is None or _hemolysis_predictor_path != model_path:
        _hemolysis_predictor = HemolysisPredictor(model_path)
        _hemolysis_predictor_path = model_path
    return _hemolysis_predictor
