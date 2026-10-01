# Reproducing the submitted run

Everything in `generate/` was produced by the commands below, in this
repository, from the committed checkpoints and data. There is no hidden step, no
network call and no credential.

## 1. Requirements

- `git`
- [`uv`](https://docs.astral.sh/uv/) 0.12 or later — it installs the Python
  interpreter itself, so no system Python is needed
- about 2 GB of disk for the virtual environment, and 3 GB of RAM (measured
  peak 1.9 GB)
- no GPU: the committed files were produced on CPU, and the code never calls
  `.cuda()`

## 2. Clone and install

```bash
git clone https://github.com/Ragnk95/amp-challenge-2027_Francos_Group.git
cd amp-challenge-2027_Francos_Group
uv sync
```

`uv sync` resolves against the committed `uv.lock`, so it installs the same
dependency versions the submitted files were produced with. `.python-version`
pins CPython 3.12.

## 3. Generate the library and the top 100

```bash
uv run generate
```

That is the whole submission. The command takes no arguments because its
defaults *are* the submitted configuration:

| flag | default | meaning |
|---|---|---|
| `--n-sequences` | 50000 | library size |
| `--top-k` | 100 | size of the selected list |
| `--length` | 50 | maximum peptide length; the sampler draws lengths below it |
| `--seed` | 42 | the only source of randomness |

It writes two files, relative to the current directory:

```
generate/library.fasta     50,000 unique peptides
generate/top.fasta         the 100 selected candidates, in rank order
```

Measured wall clock for this command on the host in section 7: **1 h 01 min
38 s**, at 100% of one core and a peak of 1.9 GB resident. It is
single-threaded and CPU-bound, so a faster core finishes sooner and extra cores
do not help. The challenge's own validator, which generates the library twice
and also runs the pairwise similarity checks, took 3 h 16 min end to end on the
same host.

Two messages on stderr are expected and harmless:

- `QuantumVQVAE: ESM unavailable (No module named 'esm'); using embedding
  fallback (dim=128)` — the encoder half of the autoencoder is not used at
  generation time, so its ESM dependency is not installed. Only the decoder runs.
- `decoder loaded; encoder-side tensor(s) skipped: _q_weights,
  pre_encoder.weight` — the same thing, stated from the checkpoint's side.

A `PennyLaneDeprecationWarning` about NumPy 1.26 is also expected. NumPy is
pinned below 2.0 on purpose, because that is the version the checkpoints and the
committed output were produced with.

## 4. Check you got the submitted files

```bash
sha256sum generate/library.fasta generate/top.fasta
```

```
d4599b5c29110065db9a9d7dc981c4ac9ee3459a0f35375f1a0d7c871aed97a4  generate/library.fasta
0b843fb267fa83257d9daaff4bc8be344ed15cc66cacdaef555ca495c66f9eb2  generate/top.fasta
```

These must match byte for byte. Three independent runs of the command on
separate copies of the repository have produced them, one of which was the
clone the challenge validator made from GitHub. If yours do not match, the
environment differs from section 7 below; the FASTA files in the repository are
the submission.

## 5. Regenerate the score tables

The entry point writes only the two FASTA files the challenge asks for, so the
numbers behind the ordering are not recoverable from its output. Two scripts
write them out. Both are deterministic and both read `generate/library.fasta`,
so run them after step 3.

```bash
uv run python scripts/export_ranking.py     # generate/ranking_top100.csv, rewrites top.fasta
uv run python scripts/wetlab_panel.py       # generate/wetlab_panel_top100.csv
```

```
1a735b092dc7b394c870b8249c62ae6c78d6ccb977b9c831f2217cffa2814504  generate/ranking_top100.csv
9737fd0bb0f5e402830363c3f1c6bd9c1fa035450e223ada8d8ade46c07b3548  generate/wetlab_panel_top100.csv
```

`export_ranking.py` re-runs the ranking and the diversity walk independently of
the entry point and writes `top.fasta` again; it must reproduce the same 100
sequences in the same order, which is a second, independent check on step 4.
Add `--full` to score all 50,000 instead of the top 100.

`wetlab_panel.py` converts the predicted MIC from ug/mL to uM, applies the
challenge's 16 uM potency threshold, and derives MIC50, MIC90, HC50 and the
safety window. It covers the 12 of the 20 panel strains our regressor was fitted
for and names the other eight as uncovered rather than averaging over them.

## 6. Run the challenge's own validator

`scripts/verify_submission.py` is the challenge's script, unmodified. It clones
the repository from GitHub into a fresh directory, installs, generates, checks
the library and the top list, checks overlap and similarity against
`data/antibacterial.fasta`, and generates a second time to confirm the output is
identical. It needs two packages the submission itself does not:

```bash
uv run --with Levenshtein --with biopython \
  python scripts/verify_submission.py \
  https://github.com/Ragnk95/amp-challenge-2027_Francos_Group \
  --antibacterial-fasta data/antibacterial.fasta
```

It generates the library twice, so allow about three hours. The last line on
success is:

```
All checks passed. Submission is valid!
```

Run it from a directory outside the clone, or pass `--dir` somewhere else: it
creates its own copy of the repository and will not overwrite yours.

## 7. The environment the committed files were produced in

Byte-identical output is guaranteed within one environment, not across
arbitrary ones. The committed files come from:

| | |
|---|---|
| OS | Linux 7.0.0-31-generic, glibc 2.43, x86-64 |
| Python | 3.12.14 |
| uv | 0.12.17 |
| numpy | 1.26.4 |
| torch | 2.14.0 |
| pennylane | 0.44.1 |
| device | CPU |

`uv.lock` pins all of these except the OS and the interpreter patch level.
PyTorch CPU kernels for the operations used here are deterministic, so a
different Linux x86-64 host with the same locked versions reproduces the same
bytes; a different architecture or a different torch minor version may not.

## 8. Why the output is deterministic

Four sources of randomness, all pinned:

1. **Sampling.** NumPy draws from a `Generator` seeded with `--seed`, including
   both the Born-distribution draw and the length draw.
2. **Decoding.** Each decoder call is preceded by its own `torch.manual_seed`,
   derived from the run seed and the call index as
   `(seed * 1_000_003 + call) % (2**31 - 1)`, so the sequence of decodes does
   not depend on how many calls preceded it within a batch.
3. **Set iteration.** `PYTHONHASHSEED` is set to `0` at import time, before any
   set is iterated. This is not decoration: string hashing is randomised per
   process in Python, and a set iterated in hash order had already changed
   top-100 membership between two runs of this group's internal pipeline.
4. **Ordering.** The ranking sorts on `(-rank_score, sequence)`. The sequence
   itself is the tie-break, so equal scores never resolve by insertion or
   dictionary order.

## 9. Checksums of the committed inputs

If a regeneration diverges, check the inputs first.

```
86c307e69449e6f13130bd96e4b7ccfb5b021f0e62ef37618a74f709f0b680fb  checkpoint/qcbm_prior.pt
2eb9400b5d983dfb3766d91f5b89e3eb7964157f558f211c97fe03e4d79fb007  checkpoint/quantum_vqvae_ep0015.pt
cbbeac64ba95746d87961e8ad9dd0849ae8058d15a300b2e7f6990730ca521e9  data/antibacterial.fasta
5d41a486b15aebea11cef61087dd99c3d4c7d0325e8bedff4cf0b05cb0c653cc  data/external/dbaasp_mic_regressor.json
801c8f9428f5432613d4a1ade8c92c126161f370f7230ab67421bc934041bdac  data/external/hemolysis_lr_model.json
3b8d4f9859caaf622265cb225ef0689b3b858e28209527cf4f2786b31cd4c13f  data/training/nr80_reference.csv
b1208b7e45b7268a2d6eaa59d7727019116121cac851fbdae65ed919479d4205  data/training/qcbm_training_sequences.fasta
```

## 10. Changing the run

The four flags in section 3 are the only knobs, and changing any of them changes
the output. `--seed 43` gives a different, equally valid library;
`--n-sequences 200` gives a quick smoke test that exercises every code path in
about a minute. Nothing else about the method is configurable at the command
line, by design: the thresholds, the scoring weights and the masked residues are
in the source, so a reviewer reads them rather than reconstructing them from a
command history.
