# AMP Challenge 2027 — Francos Group

A quantum circuit Born machine, used as the prior over the binary latent space of
a peptide autoencoder, samples the library. Both checkpoints in `checkpoint/` were
trained for this work.

```bash
uv sync
uv run generate                 # writes generate/library.fasta and generate/top.fasta
```

The defaults are the submitted configuration: 50,000 sequences, top 100, maximum
length 50, seed 42. [`REPRODUCE.md`](REPRODUCE.md) gives the exact steps, the
expected SHA-256 of every generated file, and the environment the committed files
were produced in.

## Abstract

The abstract for this submission is [`ABSTRACT.md`](ABSTRACT.md). It describes
the method, what the submitted run produced, the training data, the external
resources, every filter threshold, and the statement on manual intervention.
Everything it describes is in this repository and runs from `uv run generate`.

What follows here is the shorter description of the entry point itself.

Antimicrobial peptides are cationic and amphipathic, and the search space is
large enough that the sampling distribution matters more than the filter applied
afterwards. This submission draws from a 12-qubit parameterised quantum circuit
(ry-rz-cnot-ring, three layers) whose Born distribution over 4096 binary codes is
the generative model; a WaveNet decoder turns a sampled code into a peptide. The
circuit is not a re-parameterisation of a classical sampler: it was trained on
the binary codes of an autoencoder fitted to known antimicrobial peptides, and
against that 3000-code corpus it sits 15.4% closer in KL divergence than a
uniform proposal over the same space (1.8489 against 2.1867 bits, Born entropy
11.65 of 12 bits).

That figure is stated because it is the honest size of the effect. The prior is
informative and the advantage is moderate; nothing here rests on the word
"quantum" doing work that the measurement does not support.

Candidates are ranked by a six-term composite covering membrane insertion,
amphipathicity, net charge against a length-dependent optimum, protease
stability, helix propensity and sequence complexity, plus a MIC prior added at a
quarter weight.

That prior is a ridge regression on fifteen sequence descriptors, trained on 2900
experimental measurements from DBAASP, cross-validated RMSE 0.590 in log10, which
is a factor of about 3.9 in MIC. It ranks candidates; it does not measure them,
and no potency claim here rests on it. It is added to the composite rather than
replacing it because the model's own recorded scope says to use it as an additive
signal, not as a stand-alone potency estimate, until it has external validation.

The descriptor functions are the research pipeline's own modules, copied unchanged
into `src/amp_challenger/` rather than re-derived. An earlier draft of this entry
point rewrote them from the published formulas and three of six disagreed with the
originals, protease stability by a full unit because the cleavage set had been
widened beyond the aromatics. The MIC model is a regression over exactly those
descriptors, so a descriptor that drifts does not give a slightly worse
prediction, it gives a meaningless one.

## What the submitted run produced

Measured on the files in `generate/`, which are the output of the command above.

| quantity | value |
|---|---|
| library | 50,000 unique peptides, 8 to 49 residues, mean 21.6 |
| library sequences containing C, D or E | 0 |
| library sequences identical to a `data/antibacterial.fasta` entry | 0 |
| top 100 | 17 to 41 residues, mean 27.9; net charge +6.00 to +10.10, mean +8.89 |
| maximum identity of a selected peptide to the 39,448 references | 0.476 |
| maximum pairwise identity within the top 100 | 0.613 |
| median predicted *E. coli* MIC of the top 100 | 9.84 ug/mL |

`generate/ranking_top100.csv` carries one row per selected peptide with every
term of the composite, the MIC predictions and the two identity margins that
decided admission. `generate/wetlab_panel_top100.csv` converts the predictions
to uM against the wet-lab panel.

## Training data

- The autoencoder and the Born machine were trained on antimicrobial peptides
  drawn from DBAASP and from the APD/DRAMP families of public AMP databases,
  reduced at 80% sequence identity before training.
- No sequence from the challenge's `data/antibacterial.fasta` is emitted: it is
  loaded at generation time and any exact match is rejected by the compliance
  gate. Zero of the 50,000 submitted sequences appear in that file.

## External databases and filters

| Stage | Rule |
| --- | --- |
| Alphabet | the 20 standard amino acids only |
| Length | 8 to 50 residues |
| Duplicates | removed within the library |
| Known antibacterials | exact matches against `data/antibacterial.fasta` rejected |
| Complexity | Shannon entropy over residues at or above 0.35, which removes homopolymers |
| Top 100 | at most 80% Levenshtein identity to any reference in `data/antibacterial.fasta`, and at most 80% to any peptide already selected |

## Selection and ranking procedure

Every step below is executed by `uv run generate`; none of it is done by hand.

1. **Sample.** A code is drawn from the QCBM's Born distribution over the 4096
   binary latent codes. Length is drawn from a Gaussian centred at 16 residues
   with standard deviation 8, truncated to the allowed 8 to 50.
2. **Decode.** A WaveNet decoder turns the code into a peptide. Cysteine,
   aspartate and glutamate are masked at the logit level, so they are never
   generated rather than generated and discarded.
3. **Gate.** A candidate is kept only if it uses the 20 standard amino acids, is
   8 to 50 residues, is not already in the library, is not identical to any
   sequence in `data/antibacterial.fasta`, carries at least three K or R, has no
   single residue above 40% of its length, repeats no 6-mer within itself, and
   has residue entropy at or above 0.35.
4. **Repeat** until the library holds exactly the requested number of unique
   sequences.
5. **Rank.** Each sequence is scored as `0.75 x composite + 0.25 x MIC term`.
   The composite is the project's five-term score over membrane insertion,
   amphipathicity, net charge against a length-dependent optimum, protease
   stability, helix propensity and complexity. The MIC term folds the predicted
   Gram-negative MIC into [0, 1], with 3 ug/mL mapping to 1 and 316 ug/mL to 0.
   Ties are broken on the sequence itself, so ordering never depends on
   dictionary iteration.
6. **Filter the top list.** Walking the ranking in order, a candidate is skipped
   if it exceeds 80% Levenshtein identity against any of the 39,448 references,
   or against any peptide already selected. The first 100 survivors are the
   submission.

## Manual intervention

None. No sequence was hand-picked, hand-edited, hand-removed or reordered. The
library and the top 100 are whatever the command above produces from the seed,
and two runs of it produce byte-identical files.

The human decisions are all upstream of the run and are visible in the code: the
choice of generator, the terms in the composite and their weights, the
thresholds in the gate, and the decision to add the MIC prior at a quarter weight
rather than let it replace the composite.

## Reproducibility

The full procedure, with checksums and the environment, is in
[`REPRODUCE.md`](REPRODUCE.md). The challenge's own validator
(`scripts/verify_submission.py`, unmodified) has been run against this
repository as published and reports `All checks passed. Submission is valid!`

`uv run generate` twice produces byte-identical `library.fasta` and `top.fasta`.
Every source of randomness is pinned: NumPy draws from a seeded `Generator`, each
decoder call is preceded by its own `torch` seed derived from the run seed and the
call index, and `PYTHONHASHSEED` is fixed at import time. That last one is not
decoration: string hashing is randomised per process in Python, and a set iterated
in hash order had already changed top-100 membership between two runs of this
group's pipeline.

## Scope

This repository is the complete submitted method. It needs no API credential, no
network access and no third-party weight file, and every number in `generate/` and
in `ABSTRACT.md` came out of the command above.

It is not the whole of the group's research programme, and deliberately does not
describe it. That programme uses third-party predictors of 0.9 and 1.3 GB under
their own terms and a generative structure prediction reached over a personal API
credential, which returns a different value for the same peptide on a later call.
Neither can sit inside a submission that must reproduce byte for byte, so neither
is here, and no figure produced with them is quoted anywhere in this repository.
Helix propensity is taken from the deterministic Chou-Fasman table instead.

## Layout

```
ABSTRACT.md                          the abstract, the run's results, data and filters
REPRODUCE.md                         exact steps, checksums, environment
checkpoint/quantum_vqvae_ep0015.pt   peptide autoencoder, 12-bit binary latent
checkpoint/qcbm_prior.pt             Born machine over those codes
data/antibacterial.fasta             challenge reference, used as an exclusion filter
data/training/                       the sequences the two checkpoints were fitted to
data/external/                       the fitted MIC and haemolysis coefficients
generate/library.fasta               the submitted 50,000
generate/top.fasta                   the submitted 100, in rank order
generate/ranking_top100.csv          every score behind the ordering
generate/wetlab_panel_top100.csv     the same 100 against the wet-lab panel, in uM
src/amp_challenge_2027/generate.py   entry point
src/amp_challenger/                  scoring and compliance modules
src/peptide_gen/                     model definitions for the two checkpoints
scripts/export_ranking.py            rebuilds the ranking tables from the library
scripts/wetlab_panel.py              rebuilds the panel table
scripts/verify_submission.py         the challenge validator, unmodified
```

## Licence

MIT, see `LICENSE`.
