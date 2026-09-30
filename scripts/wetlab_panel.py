#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Score the submitted top 100 against the wet-lab panel.

The challenge scores MIC in micromolar against 20 strains, with a potency
threshold of 16 uM, and a safety window of HC50 divided by MIC50. Our MIC
regressor predicts in ug/mL and covers 12 of those 20 strains, so this converts
units, reports per-category metrics over the covered subset only, and says so in
the output rather than presenting a 12-strain result as a 20-strain one.

Writes generate/wetlab_panel_top100.csv and prints the category summary.
"""
from __future__ import annotations

import csv
import os
import statistics as st
import sys
from pathlib import Path

os.environ.setdefault("PYTHONHASHSEED", "0")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from amp_challenger.amp_scoring import HemolysisPredictor  # noqa: E402
from amp_challenger.dbaasp_mic_regressor import DBAASPMICRegressor  # noqa: E402

# Average residue masses (Da); peptide MW = sum(residues) - (n-1)*18.02 + 18.02
_AA_MASS = {
    "A": 89.09, "R": 174.20, "N": 132.12, "D": 133.10, "C": 121.16,
    "Q": 146.15, "E": 147.13, "G": 75.07, "H": 155.16, "I": 131.17,
    "L": 131.17, "K": 146.19, "M": 149.21, "F": 165.19, "P": 115.13,
    "S": 105.09, "T": 119.12, "W": 204.23, "Y": 181.19, "V": 117.15,
}

# Panel strain -> the target our regressor was fitted for, or None when we have
# no model for it. Naming the gaps is the point: a success rate computed over
# the strains we happen to cover is not a success rate over the panel.
PANEL = [
    # (strain, category, our target or None)
    ("A. baumannii ATCC 19606",        "gram_neg", "abaumannii_19606"),
    ("A. baumannii BAA-1605 (MDR)",    "gram_neg", None),
    ("E. cloacae ATCC 13047",          "gram_neg", None),
    ("E. coli ATCC 11775",             "gram_neg", "ecoli_atcc11775"),
    ("E. coli AIC221",                 "gram_neg", None),
    ("E. coli AIC222 (CRE, MDR)",      "gram_neg", None),
    ("E. coli BAA-3170 (CRE, MDR)",    "gram_neg", None),
    ("E. coli K-12 BW25113",           "gram_neg", "ecoli_k12"),
    ("K. pneumoniae ATCC 13883",       "gram_neg", "kpneumoniae_13883"),
    ("K. pneumoniae BAA-2342 (MDR)",   "gram_neg", None),
    ("P. aeruginosa PAO1",             "gram_neg", "paeruginosa_pao1"),
    ("P. aeruginosa PA14",             "gram_neg", "paeruginosa_pa14"),
    ("P. aeruginosa BAA-3197 (MDR)",   "gram_neg", None),
    ("S. enterica ATCC 9150",          "gram_neg", "senterica"),
    ("S. enterica Typhimurium 700720", "gram_neg", "senterica_typhimurium"),
    ("B. subtilis ATCC 23857",         "gram_pos", None),
    ("S. aureus ATCC 12600",           "gram_pos", "saureus_atcc12600"),
    ("S. aureus BAA-1556 (MRSA, MDR)", "gram_pos", "saureus_mrsa"),
    ("E. faecalis 700802 (VRE, MDR)",  "gram_pos", "vre_faecalis"),
    ("E. faecium 700221 (VRE, MDR)",   "gram_pos", "vre_faecium"),
]
MDR = {s for s, _, _ in PANEL if "MDR" in s}
THRESHOLD_UM = 16.0
HC50_LIMIT = 128.0


def mol_weight(seq: str) -> float:
    return sum(_AA_MASS.get(a, 110.0) for a in seq) - (len(seq) - 1) * 18.02


def ugml_to_um(mic_ugml: float, seq: str) -> float:
    return mic_ugml / mol_weight(seq) * 1000.0


def read_fasta(path: Path) -> list[str]:
    return [l.strip() for l in open(path, encoding="utf-8")
            if l.strip() and not l.startswith(">")]


def main() -> None:
    top = read_fasta(Path(sys.argv[1] if len(sys.argv) > 1 else "generate/top.fasta"))
    reg = DBAASPMICRegressor()
    hemo = HemolysisPredictor()
    covered = [(s, c, t) for s, c, t in PANEL if t]
    print("panel strains: %d; predicted by our models: %d" % (len(PANEL), len(covered)))
    print("potency threshold %.0f uM; HC50 assay limit %.0f uM" % (THRESHOLD_UM, HC50_LIMIT))
    print()

    cols = ["rank", "sequence", "length", "mw_da", "hc50_um", "safety_window"]
    for name, _, _ in covered:
        cols.append("mic_um__" + name.replace(" ", "_").replace(".", ""))
    cols += ["success_overall", "success_gram_neg", "success_gram_pos", "success_mdr",
             "mic50_um", "mic90_um"]

    rows, summary = [], []
    for i, seq in enumerate(top, start=1):
        mw = mol_weight(seq)
        mics: dict[str, float] = {}
        for name, cat, tgt in covered:
            p = reg.predict(seq, tgt)
            mics[name] = ugml_to_um(float(p.mic_ugml), seq) if p else float("nan")
        vals = [v for v in mics.values() if v == v]
        try:
            hc50 = float(hemo.predict_hc50_um(seq))
        except Exception:
            hc50 = float("nan")
        mic50 = st.median(vals) if vals else float("nan")
        mic90 = sorted(vals)[max(0, int(0.9 * len(vals)) - 1)] if vals else float("nan")

        def rate(sel) -> float:
            v = [mics[n] for n, c, _ in covered if sel(n, c) and mics[n] == mics[n]]
            return 100.0 * sum(1 for x in v if x <= THRESHOLD_UM) / len(v) if v else float("nan")

        row = {
            "rank": i, "sequence": seq, "length": len(seq), "mw_da": round(mw, 1),
            "hc50_um": round(hc50, 2), "mic50_um": round(mic50, 2), "mic90_um": round(mic90, 2),
            "safety_window": round(hc50 / mic50, 2) if mic50 and mic50 == mic50 else float("nan"),
            "success_overall": round(rate(lambda n, c: True), 1),
            "success_gram_neg": round(rate(lambda n, c: c == "gram_neg"), 1),
            "success_gram_pos": round(rate(lambda n, c: c == "gram_pos"), 1),
            "success_mdr": round(rate(lambda n, c: n in MDR), 1),
        }
        for name, _, _ in covered:
            row["mic_um__" + name.replace(" ", "_").replace(".", "")] = round(mics[name], 2)
        rows.append(row)
        summary.append((i, seq, row["success_overall"], row["success_gram_neg"],
                        row["success_gram_pos"], row["success_mdr"],
                        row["mic50_um"], row["hc50_um"], row["safety_window"]))

    out = Path("generate/wetlab_panel_top100.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print("wrote %s" % out)
    print()
    print("%-4s %-30s %7s %7s %7s %7s %8s %8s %7s"
          % ("rank", "sequence", "overall", "gram-", "gram+", "MDR", "MIC50", "HC50", "SW"))
    for r in summary[:15]:
        print("%-4d %-30s %6.0f%% %6.0f%% %6.0f%% %6.0f%% %8.1f %8.1f %7.1f"
              % (r[0], r[1][:30], r[2], r[3], r[4], r[5], r[6], r[7], r[8]))
    print()
    ov = [r[2] for r in summary]
    sw = [r[8] for r in summary if r[8] == r[8]]
    print("across the 100: median overall success %.0f%%, median safety window %.1f"
          % (st.median(ov), st.median(sw)))
    print("candidates with 100%% predicted success on the covered strains: %d"
          % sum(1 for x in ov if x >= 100.0))


if __name__ == "__main__":
    main()
