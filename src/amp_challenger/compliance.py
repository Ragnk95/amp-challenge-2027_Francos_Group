"""AMP sequence compliance checker and diversity filter for challenge submission.

Validates:
- Standard 20-AA alphabet only (no ambiguous/modified residues)
- Length bounds (8–50 aa)
- Duplicate detection (first-occurrence wins)
- Max single-AA frequency ≤ 40% (prevents homopolymers and repeat-motif inflation)
- No repeated k-mer ≥ 6 aa (rejects tandem/low-complexity artifacts the 40% cap misses)
- Minimum cationic residues (K+R ≥ 3) — essential AMP property
- No anionic residues (D, E) — negatively charged AAs reduce membrane activity
- No cysteine (C) — disulfide bridges introduce non-linear structure
"""
from __future__ import annotations

from collections import Counter

VALID_AMP_ALPHABET_STR: str = "ACDEFGHIKLMNPQRSTVWY"
VALID_AMP_ALPHABET: frozenset[str] = frozenset(VALID_AMP_ALPHABET_STR)

# Biological filters (from MRL notebook analysis)
MAX_SINGLE_AA_FREQ: float = 0.40   # no AA may exceed 40% of the sequence
MIN_CATIONIC_COUNT: int = 3        # minimum K + R residues (absolute)
MIN_CATIONIC_FRACTION: float = 0.0   # fractional K/R constraint disabled — absolute count (≥3) is sufficient
# Longest substring allowed to recur within a single sequence.  A k-mer of
# length > this repeating in-sequence (e.g. GLMQFIKR…GLMQFIKR) signals a
# tandem-repeat / low-complexity artifact that the 40% single-AA cap misses
# because no individual residue is over-represented.  k=6 catches every such
# artifact observed in generated top-100s while flagging only ~4.5% of natural
# AMPs (calibrated on all_amps_indep_nr80, 2026-06-23).
MAX_REPEATED_KMER_LEN: int = 5

# Amino acids excluded from AMP candidates:
#   C (cysteine)   — disulfide bridges → non-linear structure
#   D (aspartate)  — anionic → reduces net cationic charge and membrane binding
#   E (glutamate)  — anionic → same reason as D
FORBIDDEN_AAS: frozenset[str] = frozenset("CDE")


def has_low_complexity_repeat(seq: str, kmer_len: int = MAX_REPEATED_KMER_LEN + 1) -> bool:
    """Return True when any substring of ``kmer_len`` aa recurs in the sequence.

    Detects dispersed tandem repeats (e.g. ``GLMQFIKR…GLMQFIKR``) that the
    single-AA frequency cap cannot catch.  A repeated 6-mer also implies any
    longer shared motif, so checking one length is sufficient.  O(n) for
    peptides ≤ 50 aa.
    """
    seen: set[str] = set()
    for i in range(len(seq) - kmer_len + 1):
        kmer = seq[i:i + kmer_len]
        if kmer in seen:
            return True
        seen.add(kmer)
    return False


def _has_low_complexity_repeat(seq: str, kmer_len: int = MAX_REPEATED_KMER_LEN + 1) -> bool:
    """Backward-compatible private alias for older internal callers."""
    return has_low_complexity_repeat(seq, kmer_len)


def _passes_bio_filters(seq: str, min_kr_fraction: float = MIN_CATIONIC_FRACTION) -> bool:
    """Check forbidden residues, max single-AA frequency and minimum cationic requirements."""
    n = len(seq)
    counts = Counter(seq)
    # Reject sequences containing any forbidden amino acid (C, D, E)
    if any(counts.get(aa, 0) > 0 for aa in FORBIDDEN_AAS):
        return False
    # any single amino acid > 40% → reject
    if counts.most_common(1)[0][1] / n > MAX_SINGLE_AA_FREQ:
        return False
    # repeated ≥6-mer → tandem/low-complexity artifact → reject
    if _has_low_complexity_repeat(seq):
        return False
    kr = counts.get("K", 0) + counts.get("R", 0)
    # absolute minimum: at least 3 cationic residues
    if kr < MIN_CATIONIC_COUNT:
        return False
    # fractional minimum: only applied when the caller explicitly sets a
    # non-zero threshold (config.min_kr_fraction).  The default (0.0) keeps
    # the absolute-count-only behaviour so existing filters are not affected.
    if min_kr_fraction > 0.0 and kr / n < min_kr_fraction:
        return False
    return True


def compliance_failures(
    sequence: str,
    min_len: int = 8,
    max_len: int = 30,
    min_kr_fraction: float = MIN_CATIONIC_FRACTION,
) -> tuple[str, ...]:
    """Return stable reason codes explaining why ``sequence`` is non-compliant.

    The reason names match ``compliance_report()`` keys so generator-level
    diagnostics can be aggregated without guessing which rule failed.
    """
    seq = sequence.upper().strip()
    n = len(seq)
    reasons: list[str] = []
    if not all(aa in VALID_AMP_ALPHABET for aa in seq):
        reasons.append("invalid_alphabet")
    if n < min_len:
        reasons.append("too_short")
    if n > max_len:
        reasons.append("too_long")
    if n == 0:
        return tuple(reasons)

    counts = Counter(seq)
    if counts.most_common(1)[0][1] / n > MAX_SINGLE_AA_FREQ:
        reasons.append("high_aa_freq")
    kr = counts.get("K", 0) + counts.get("R", 0)
    if kr < MIN_CATIONIC_COUNT or (
        min_kr_fraction > 0.0 and kr / n < min_kr_fraction
    ):
        reasons.append("low_cationic")
    if any(counts.get(aa, 0) > 0 for aa in FORBIDDEN_AAS):
        reasons.append("forbidden_aa")
    if has_low_complexity_repeat(seq):
        reasons.append("low_complexity")
    return tuple(reasons)


def is_compliant(
    sequence: str,
    min_len: int = 8,
    max_len: int = 30,
    min_kr_fraction: float = MIN_CATIONIC_FRACTION,
) -> bool:
    """Return True when sequence passes all compliance checks."""
    seq = sequence.upper().strip()
    n = len(seq)
    if n < min_len or n > max_len:
        return False
    if not all(aa in VALID_AMP_ALPHABET for aa in seq):
        return False
    return _passes_bio_filters(seq, min_kr_fraction)


def filter_compliant(
    sequences: list[str],
    min_len: int = 8,
    max_len: int = 30,
    min_kr_fraction: float = MIN_CATIONIC_FRACTION,
) -> list[str]:
    """Return compliant, deduplicated sequences in first-occurrence order."""
    seen: set[str] = set()
    result: list[str] = []
    for seq in sequences:
        norm = seq.upper().strip()
        if norm not in seen and is_compliant(norm, min_len, max_len, min_kr_fraction):
            seen.add(norm)
            result.append(norm)
    return result


def compliance_report(
    sequences: list[str],
    min_len: int = 8,
    max_len: int = 30,
    min_kr_fraction: float = MIN_CATIONIC_FRACTION,
) -> dict[str, int]:
    """Return a summary dict of compliance statistics for a sequence list."""
    total = len(sequences)
    normed = [s.upper().strip() for s in sequences]
    # Single pass: pre-compute Counter and length once per sequence so the
    # downstream checks don't each recompute Counter(s).
    counts_and_len: list[tuple[Counter[str], int]] = [(Counter(s), len(s)) for s in normed]

    invalid_alphabet = 0
    too_short = 0
    too_long = 0
    high_freq = 0
    low_cation = 0
    forbidden_aa = 0
    low_complexity = 0
    passing = 0
    for s, (cnt, n) in zip(normed, counts_and_len):
        bad_alphabet = not all(aa in VALID_AMP_ALPHABET for aa in s)
        if bad_alphabet:
            invalid_alphabet += 1
        if n < min_len:
            too_short += 1
            continue
        if n > max_len:
            too_long += 1
        hi = cnt.most_common(1)[0][1] / n > MAX_SINGLE_AA_FREQ
        if hi:
            high_freq += 1
        kr = cnt.get("K", 0) + cnt.get("R", 0)
        low = kr < MIN_CATIONIC_COUNT or kr / n < min_kr_fraction
        if low:
            low_cation += 1
        forb = any(cnt.get(aa, 0) > 0 for aa in FORBIDDEN_AAS)
        if forb:
            forbidden_aa += 1
        lc = _has_low_complexity_repeat(s)
        if lc:
            low_complexity += 1
        # Derive `passing` from the already-collected booleans rather than
        # re-running is_compliant() (which would rebuild Counter and re-scan
        # the sequence a second time — doubling compliance cost on large
        # libraries).
        if not (bad_alphabet or n > max_len or hi or low or forb or lc):
            passing += 1
    duplicates = total - len(set(normed))

    return {
        "total":            total,
        "passing":          passing,
        "invalid_alphabet": invalid_alphabet,
        "too_short":        too_short,
        "too_long":         too_long,
        "duplicates":       duplicates,
        "high_aa_freq":     high_freq,
        "low_cationic":     low_cation,
        "forbidden_aa":     forbidden_aa,
        "low_complexity":   low_complexity,
    }


def _seq_similarity(a: str, b: str) -> float:
    """Normalised Levenshtein similarity in [0, 1].

    similarity = 1 − edit_distance / max(len(a), len(b))

    Preferred over SequenceMatcher.ratio() because it reflects actual
    mutational distance between peptide sequences (each edit = one
    substitution/insertion/deletion) rather than longest-common-subsequence
    overlap, which underestimates similarity for shifted or repeated motifs.
    O(m × n) per pair; fast for peptides ≤ 50 aa.
    """
    m, n = len(a), len(b)
    if m == 0 and n == 0:
        return 1.0
    if m == 0 or n == 0:
        return 0.0
    # Single-row DP to compute edit distance in O(min(m,n)) space
    if m < n:
        a, b, m, n = b, a, n, m
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            temp = dp[j]
            dp[j] = prev if a[i - 1] == b[j - 1] else 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return 1.0 - dp[n] / m  # m = max(original m, n)


def prefix_diversity_filter(
    sequences: list[str],
    max_per_prefix: int = 10,
    prefix_len: int = 5,
) -> list[str]:
    """Cap the number of sequences that share the same N-terminal prefix.

    Prevents seed-family over-representation when the same motif (e.g.
    ``ILPWK…``, ``GIGKF…``) dominates the top-100 via random mutations.
    Iterates in order (assumes sequences are already ranked best-first).
    """
    counts: dict[str, int] = {}
    result: list[str] = []
    for seq in sequences:
        prefix = seq[:prefix_len]
        if counts.get(prefix, 0) < max_per_prefix:
            counts[prefix] = counts.get(prefix, 0) + 1
            result.append(seq)
    return result


def diversity_filter(
    sequences: list[str],
    max_similarity: float = 0.95,
    n: int | None = None,
) -> list[str]:
    """Greedy diversity filter — keep sequences that are sufficiently different.

    Uses Levenshtein similarity (1 − edit_distance / max_length) as the
    distance metric.  Iterates sequences in order (assumes they are already
    ranked best-first).  A sequence is kept only if its Levenshtein similarity
    to every already-kept sequence is below ``max_similarity``.
    Runs in O(kept × total) — suitable for sets up to ~10k sequences.

    Parameters
    ----------
    sequences:
        Pre-ranked sequences (best first).
    max_similarity:
        Maximum allowed pairwise similarity (default 0.95 = 95%).
    n:
        If set, stop once ``n`` sequences have been selected.
    """
    kept: list[str] = []
    for seq in sequences:
        if n is not None and len(kept) >= n:
            break
        if all(_seq_similarity(seq, k) < max_similarity for k in kept):
            kept.append(seq)
    return kept
