# AMP Challenge 2027 — Francos Group

## Abstract

`uv run generate` produces the submitted library and top 100. It samples a
12-qubit quantum circuit Born machine (ry-rz-cnot-ring, three layers) used as the
prior over the binary latent space of a peptide autoencoder: a code is drawn from
the Born distribution over 4096 states and a WaveNet decoder turns it into a
peptide, with cysteine, aspartate and glutamate masked at the logit level so they
are never generated. Against the 3000-code training corpus the prior sits 15.4%
closer in KL divergence than a uniform proposal over the same space (1.8489
against 2.1867 bits, Born entropy 11.65 of 12).

Candidates pass a compliance gate (20 standard amino acids, 8 to 50 residues, no
duplicates, nothing identical to a known antibacterial, at least three K or R, no
residue above 40% of the sequence, no repeated 6-mer, residue entropy at least
0.35), and the library is filled to exactly 50,000 unique peptides. Ranking is a
six-term physicochemical composite (Wimley-White membrane insertion, Eisenberg
hydrophobic moment, net charge against a length-dependent optimum, protease
stability, helix propensity, sequence complexity) plus a DBAASP-trained MIC prior
at a quarter weight, with ties broken on the sequence itself. The top 100 is then
filtered to at most 80% Levenshtein identity against every reference and against
every peptide already selected.

Every checkpoint this needs is in `checkpoint/`, every filter runs from committed
data, and two runs of the command produce byte-identical FASTA files.

## What the submitted run produced

Every number here was measured on the files in `generate/`, which are the output
of the command above at its default settings. Nothing in this document describes
any other run.

| quantity | value |
|---|---|
| library size | 50,000, all unique |
| library length | 8 to 49 residues, mean 21.6 |
| library sequences containing C, D or E | 0 |
| library sequences identical to a `data/antibacterial.fasta` entry | 0 |
| top-100 length | 17 to 41 residues, mean 27.9 |
| top-100 net charge | +6.00 to +10.10, mean +8.89 |
| maximum identity of any selected peptide to the 39,448 references | 0.476 |
| maximum pairwise identity within the top 100 | 0.613 |
| rank score across the top 100 | 0.6751 to 0.6990 |
| median predicted *E. coli* MIC of the top 100 | 9.84 ug/mL |

The per-peptide values behind the ranking are in
`generate/ranking_top100.csv`, one row per selected peptide with every term of
the composite, the MIC predictions and the two identity margins that decided
admission. The wet-lab panel conversion is in
`generate/wetlab_panel_top100.csv`.

## Training data

No foundation model was trained from scratch. The two checkpoints that ship were
fitted for this work, and the sequences behind them are committed, not merely
described.

| file | rows | what it is |
|---|---|---|
| `data/training/qcbm_training_sequences.fasta` | 3,000 | the peptides whose binary latent codes the Born machine was fitted to; 8 to 30 residues |
| `data/training/nr80_reference.csv` | 2,348 | the non-redundant 80% identity reference set, with the source database and original identifier for each sequence |
| `data/antibacterial.fasta` | 39,448 | the challenge's own reference, used here only as an exclusion filter |
| `data/external/dbaasp_mic_regressor.json` | 34,587 measurements | the fitted MIC model; the DBAASP export behind it is redistributable only under DBAASP's terms, so the fitted coefficients ship and the raw export does not |
| `data/external/hemolysis_lr_model.json` | 985 HC50 records | logistic regression over 16 features, used only for the wet-lab panel, never for ranking |

Source databases for the generative corpus: DBAASP, DRAMP and APD3, merged,
globally deduplicated, reduced at 80 to 90% identity and filtered to 8 to 50
residues. The merge rewrites headers, so the combined file cannot attribute an
individual sequence to one database; `nr80_reference.csv` keeps that attribution
where it survives.

Two empirical calibrations were derived from sequence data rather than trained as
models. The charge optimum in the scoring function came from 1,678 peptides with
*E. coli* MIC annotations in DBAASP, stratified by length, and yields a
length-dependent optimum near 0.35 x length instead of the fixed +4 to +6
constant used in the literature. The MIC prior is a ridge regression on fifteen
sequence descriptors fitted to 2,900 DBAASP measurements, cross-validated RMSE
0.590 in log10, which is a factor of about 3.9 in MIC.

## External resources and filters

Everything the submitted run reads is in this repository. No network call, no API
credential and no third-party weight file is involved.

| resource | used for | where it comes from |
|---|---|---|
| `checkpoint/qcbm_prior.pt` | the generative prior | fitted here, 2.1 KB |
| `checkpoint/quantum_vqvae_ep0015.pt` | the decoder | fitted here, 5.0 MB |
| Wimley-White interfacial scale | membrane insertion term | published scale, in `src/amp_challenger/amp_scoring.py` |
| Eisenberg consensus scale | hydrophobic moment, normalised by the scale extreme (arginine, 2.53) | published scale, same module |
| Chou-Fasman P(alpha) | helix propensity | published 1978 table; deterministic by construction |
| `data/external/dbaasp_mic_regressor.json` | the MIC term, a quarter of the rank score | fitted here from DBAASP |
| `data/antibacterial.fasta` | exclusion filter and the 80% identity novelty gate | the challenge's own file |

Filtering runs in two layers. The compliance gate rejects outright; a candidate
is dropped if it violates any of the following.

| rule | threshold |
|---|---|
| alphabet | the 20 canonical amino acids, no X, B or Z |
| length | 8 to 50 residues |
| forbidden residues | C, D, E — masked at the decoder logits, so never generated |
| maximum single-residue frequency | 40% |
| cationic residues (K + R) | 3 or more |
| repeated k-mer | no repeated 6-mer within the sequence |
| residue entropy | at or above 0.35 |
| duplicates | removed within the library |
| identity to a known antibacterial | exact matches rejected |

The composite score then only orders what survives, and the top-100 walk applies
the diversity gate: at most 80% Levenshtein identity to any of the 39,448
references and to any peptide already selected. Identity uses a length-based
upper bound, 1 - |L1 - L2| / max(L1, L2), purely to skip pairs that cannot
mathematically reach the threshold, so no pair is discarded without being
computed unless discarding it is provable.

Predicted MIC values are applied outside the predictor's training domain. They
order candidates and are not presented as measurements.

## Manual intervention

None. No sequence was hand-picked, hand-edited, hand-removed or reordered. The
library and the top 100 are whatever `uv run generate` produces from seed 42, and
two runs of it produce byte-identical files.

Human intervention was confined to the code and is visible in it: the choice of
generator, the terms in the composite and their weights, the thresholds in the
gate, and the decision to add the MIC prior at a quarter weight rather than let
it replace the composite.

## Reproducing this

`REPRODUCE.md` gives the exact commands, the environment the committed files were
produced in, and the SHA-256 of every file a reviewer should be able to
regenerate.
