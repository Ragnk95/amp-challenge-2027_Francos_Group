#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Write the submitted top 100 as FASTA with the ranking numbers in the headers.

``generate/top.fasta`` is the submitted file and carries bare ``>seqN`` headers,
because that is the format the challenge reads and the one the published
checksums cover. This writes a second, annotated copy for anyone reading the
list by hand: same sequences, same order, with the score, its components and the
wet-lab panel values on each header line.

It reads only committed tables, and it refuses to write anything if the order it
reconstructs disagrees with ``generate/top.fasta``.

Reads  generate/top.fasta
       generate/ranking_top100.csv
       generate/wetlab_panel_top100.csv
Writes generate/top100_ranked.fasta
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("PYTHONHASHSEED", "0")

HEADER = (
    "rank{rank:03d}"
    " score={score} composite={composite} mic_term={mic_term}"
    " len={length} charge={charge:+.2f}"
    " mic_gram_neg={mic_gn}ug/mL"
    " identity_to_reference={id_ref} identity_to_selected={id_sel}"
)
PANEL = (
    " mic50={mic50}uM mic90={mic90}uM hc50={hc50}uM"
    " safety_window={sw} success_covered={success}%"
)


def read_fasta(path: Path) -> list[str]:
    return [line.strip() for line in open(path, encoding="utf-8")
            if line.strip() and not line.startswith(">")]


def read_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="generate",
                    help="directory holding top.fasta and the two score tables")
    args = ap.parse_args()
    d = Path(args.dir)

    submitted = read_fasta(d / "top.fasta")
    ranking = read_csv(d / "ranking_top100.csv")
    panel = {r["sequence"]: r for r in read_csv(d / "wetlab_panel_top100.csv")}

    if [r["sequence"] for r in ranking] != submitted:
        raise SystemExit(
            "ranking_top100.csv does not match top.fasta in sequence or order; "
            "re-run scripts/export_ranking.py before this script"
        )

    out = d / "top100_ranked.fasta"
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        for r in ranking:
            head = HEADER.format(
                rank=int(r["rank"]), score=r["rank_score"],
                composite=r["composite"], mic_term=r["mic_term"],
                length=r["length"], charge=float(r["net_charge"]),
                mic_gn=r["mic_gram_neg_ugml"],
                id_ref=r["max_identity_to_reference"],
                id_sel=r["max_identity_to_selected"],
            )
            p = panel.get(r["sequence"])
            if p:
                head += PANEL.format(
                    mic50=p["mic50_um"], mic90=p["mic90_um"], hc50=p["hc50_um"],
                    sw=p["safety_window"], success=p["success_overall"],
                )
            fh.write(f">{head}\n{r['sequence']}\n")

    if read_fasta(out) != submitted:
        raise SystemExit("the annotated file does not round-trip to top.fasta")
    print(f"{len(ranking)} records -> {out}")


if __name__ == "__main__":
    main()
