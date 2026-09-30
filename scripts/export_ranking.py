#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Rank a generated library and export the top list with every score behind it.

``generate`` writes the two FASTA files the challenge asks for and nothing else,
so the numbers that produced the ordering are not recoverable from its output.
This writes them out: one row per selected peptide, every term of the composite,
the MIC prediction, and the identity margins that decide whether a candidate was
admitted.

Reads  generate/library.fasta
Writes generate/top.fasta
       generate/ranking_top100.csv
       generate/ranking_library.csv   (--full, every sequence scored)
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTHONHASHSEED", "0")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from amp_challenge_2027 import generate as G  # noqa: E402
from amp_challenger.amp_scoring import (  # noqa: E402
    compute_amphipathicity,
    compute_complexity_score,
    compute_hydrophobicity,
    compute_membrane_insertion_score,
    compute_net_charge,
    compute_stability_proxy,
    _compute_helix_propensity_chou_fasman as compute_helix,
)
from amp_challenger.dbaasp_mic_regressor import DBAASPMICRegressor  # noqa: E402

FIELDS = [
    "rank", "sequence", "length", "net_charge",
    "rank_score", "composite", "mic_term",
    "insertion", "amphipathicity", "charge_term", "selectivity",
    "stability", "helix", "complexity", "hydrophobicity",
    "mic_gram_neg_ugml", "mic_gram_pos_ugml", "mic_ecoli_ugml",
    "max_identity_to_reference", "max_identity_to_selected",
]


def read_fasta(path: Path) -> list[str]:
    out: list[str] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith(">"):
                out.append(line.upper())
    return out


def write_fasta(seqs: list[str], path: Path) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for i, s in enumerate(seqs, start=1):
            fh.write(f">seq{i}\n{s}\n")


def mic_of(reg: DBAASPMICRegressor, seq: str, target: str) -> float:
    if not reg.is_loaded:
        return float("nan")
    p = reg.predict(seq, target)
    return float(p.mic_ugml) if p is not None else float("nan")


def row_for(reg, seq: str, position: int, ref_id: float, sel_id: float) -> dict:
    return {
        "rank": position,
        "sequence": seq,
        "length": len(seq),
        "net_charge": round(compute_net_charge(seq), 2),
        "rank_score": round(G.rank_score(seq), 6),
        "composite": round(G.composite(seq), 6),
        "mic_term": round(G.mic_term(seq), 6),
        "insertion": round(compute_membrane_insertion_score(seq), 4),
        "amphipathicity": round(compute_amphipathicity(seq), 4),
        "charge_term": round(G.charge_term(seq), 4),
        "selectivity": round(G.selectivity(seq), 4),
        "stability": round(compute_stability_proxy(seq), 4),
        "helix": round(compute_helix(seq), 4),
        "complexity": round(compute_complexity_score(seq), 4),
        "hydrophobicity": round(compute_hydrophobicity(seq), 4),
        "mic_gram_neg_ugml": round(mic_of(reg, seq, "gram_neg"), 2),
        "mic_gram_pos_ugml": round(mic_of(reg, seq, "gram_pos"), 2),
        "mic_ecoli_ugml": round(mic_of(reg, seq, "ecoli_atcc11775"), 2),
        "max_identity_to_reference": round(ref_id, 4),
        "max_identity_to_selected": round(sel_id, 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--library", default="generate/library.fasta")
    ap.add_argument("--out-dir", default="generate")
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--full", action="store_true",
                    help="also score every sequence in the library")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    library = read_fasta(Path(args.library))
    print(f"library: {len(library)} sequences")

    refs = read_fasta(G.ANTIBACTERIAL) if G.ANTIBACTERIAL.exists() else []
    print(f"references: {len(refs)}")
    reg = DBAASPMICRegressor()

    # Same ordering the entry point uses: rank_score descending, sequence as the
    # tie-break so nothing depends on iteration order.
    ranked = sorted(library, key=lambda s: (-G.rank_score(s), s))

    chosen: list[str] = []
    rows: list[dict] = []
    for cand in ranked:
        if len(chosen) >= args.top_k:
            break
        ref_id = max((G.levenshtein_ratio(cand, r) for r in refs), default=0.0)
        if ref_id > 0.8:
            continue
        sel_id = max((G.levenshtein_ratio(cand, k) for k in chosen), default=0.0)
        if sel_id > 0.8:
            continue
        chosen.append(cand)
        rows.append(row_for(reg, cand, len(chosen), ref_id, sel_id))

    write_fasta(chosen, out_dir / "top.fasta")
    with open(out_dir / "ranking_top100.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"top {len(chosen)} -> {out_dir/'top.fasta'} and ranking_top100.csv")

    if args.full:
        with open(out_dir / "ranking_library.csv", "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS)
            w.writeheader()
            for i, s in enumerate(ranked, start=1):
                w.writerow(row_for(reg, s, i, float("nan"), float("nan")))
        print(f"full library scored -> {out_dir/'ranking_library.csv'}")


if __name__ == "__main__":
    main()
