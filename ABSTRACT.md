# AMP Challenge 2027 — Francos Group

## Abstract

Antimicrobial peptides are candidates for the development of alternatives to conventional antibiotics, and generative models have expanded the possibilities for their de novo design. However, simultaneously optimizing antimicrobial activity and selectivity remains a challenge, as the physicochemical properties that favor activity can also increase toxicity. We present a pipeline that integrates six heterogeneous generators, including diffusion in the ESM latent space, parallel masked infilling with PepMLM-650M, and a VQ-VAE with a simulated variational circuit. The sequence budget is distributed among the generators at each round through Thompson sampling, using a reward based on the hypervolume contribution to a cumulative Pareto front. The two objectives considered are predicted antimicrobial activity and an estimate of selectivity defined as one minus the predicted hemolytic risk. Screening uses a five-term composite score, with an additional risk penalty associated with the membrane context. The workflow includes refinement through mutational operators, compliance filters, and diversity-based selection, with deterministic final ranking. Over 62 rounds, 50,050 sequences were evaluated, and a library of 50,000 was obtained, all of which passed the internal compliance check. Among the one hundred candidates selected after diversity filtering, the maximum pairwise identity was 0.688, and the mean maximum similarity of each candidate to the 200 seed sequences was 0.366. The minimum inhibitory concentration predicted by APEX against Escherichia coli ATCC 11775 had a median of 4.33 µM. The run also revealed near-zero hypervolume contributions among the finalists, indicating a limitation to be investigated in reward updating. The pipeline produced a library and a selection of candidates for experimental evaluation in the AMP Challenge.

## Training data, external resources and filtering

Summary of the training data, external databases and predictors, and the manual interventions and computational filters behind the submitted set of 100 peptides (run A, seed 20260929, six generator arms).

## 1. Training data

No foundation model was trained from scratch for this submission. Every generative arm uses a published pretrained checkpoint or has no learned parameters; the only model fitted in-house is a small hemolysis calibration layer.

| Component | Role | Provenance |
|---|---|---|
| CPL-Diff | ESM latent-space diffusion | Pretrained checkpoint cpl_diff_v4/ep0030, z_dim 256, 500 denoising steps |
| PepMLM-650M | Masked-language infilling | Published pretrained weights |
| QuantumVQVAE | VQ-VAE with 12-qubit variational circuit and WaveNet decoder | Frozen ESMC-600M encoder; PennyLane lightning.qubit simulator |
| LatentDiffusion | Latent diffusion | In-house hybrid framework |
| QuantumInspired, QCBM | Annealing-inspired constrained optimizers | No learned parameters |
| Hemolysis classifier | Selectivity term | Logistic regression, 16 features, fitted in-house on 985 sequences drawn from DBAASP HC50 data |

Two empirical calibrations were derived from sequence data rather than trained as models. The charge optimum of the scoring function came from 1,678 sequences with E. coli activity annotations, stratified by length; it yields a length-dependent optimum near 0.35 × length instead of the fixed +4 to +6 constant used in the literature. Generation was seeded with 200 reference sequences, which also serve as the baseline for the novelty measurement: the mean maximum similarity of the final 100 peptides to these seeds is 0.366.

## 2. External databases and predictors

All external resources are used at inference time only. None was retrained or fine-tuned on competition data.

| Resource | Used for | Notes |
|---|---|---|
| APEX, HMD-AMP | Antimicrobial activity and calibrated MIC | Target E. coli ATCC 11775. Predicted activity is one of the two Pareto objectives and also enters the final ranking |
| HemoPI2 | Hemolysis risk | Batch-cached, combined with the in-house calibrated classifier |
| ESM3-open (Forge API) | Per-residue DSSP-style secondary structure | Chou-Fasman (1978) is the documented fallback, not used in the reported run |
| Wimley–White scale | Interfacial transfer free energy | Reference range calibrated for PE/PG-rich bacterial inner membrane |
| Eisenberg scale | Hydrophobic moment | Normalized by the scale extreme (arginine, 2.53) |
| Reference AMP set (22,314 sequences) | 80% identity novelty gate | DBAASP, DRAMP, APD3 and a local positive set reduced to 90% identity, merged, globally deduplicated and filtered to 8-50 residues, plus 134 sequences from a validated-peptide set |
| deepseek-r1 (local) | Between-round objective reweighting and motif proposals | Locally hosted; never authors sequences |
| Martini 3, CHARMM36m, GROMACS 2026.3 | Molecular dynamics | Evaluated as a validator and deliberately excluded from scoring |

Reported MIC values are calibrated predictions applied outside the predictor's training domain, so they order candidates and are not presented as measurements. The molecular dynamics layer was excluded after single-replica observables proved seed-dominated; only per-residue insertion depth survived as a descriptor orthogonal to hydropathy, and it is not coupled to generation.

## 3. Computational filters

Filtering runs in three layers: two hard gates before scoring, a compliance gate that rejects outright, and a composite score that only orders what survives.

Hard gates. A candidate is voided before any term is computed if its length is outside 8–40 residues or its net charge is below +1.0. The charge gate is applied first, so an anionic peptide is never evaluated on any other axis.

Compliance gate. A candidate is rejected if it violates any of the following.

| Rule | Threshold | Rationale |
|---|---|---|
| Alphabet | 20 canonical amino acids | No X, B, Z or non-natural residues |
| Forbidden residues | C, D, E | Cysteine for disulfide and oxidation; Asp and Glu conflict with the cationic requirement |
| Max single-residue frequency | 40% | Cuts compositional collapse |
| Cationic residues (K + R) | 3 or more (absolute count) | Minimum charge carrier |
| Repeated k-mer | No repeated 6-mer | Hard rejection; the score uses a graded version of the same count |
| Identity to any known AMP | 0.80 maximum | Competition requirement |
| Pairwise identity among finalists | 0.70 maximum | Diversity filter on the submitted set |

Identity is confirmed by exhaustive alignment immediately before a candidate is accepted (match +1, mismatch −1, gap −2, normalized by alignment length); the metric was validated against an independently computed Levenshtein ratio. A length-based sound upper bound, 1 − |L1 − L2| / max(L1, L2), is used only to skip pairs that cannot mathematically reach the threshold, so no pair is discarded without alignment unless discarding it is provable. In the submitted set the maximum pairwise identity is 0.688.

Ranking. Surviving candidates are ranked by a linear composite of five normalized terms (activity, selectivity, protease stability, helix propensity, sequence complexity) minus a membrane-context risk penalty weighted at 0.10 outside the normalization. Generator budget across rounds is allocated by Thompson sampling on Pareto hypervolume contribution over two objectives, predicted activity and selectivity, rather than on the composite score itself. All 50,000 sequences in the final library passed the compliance gate.

## 4. Manual intervention

No sequence was hand-picked, hand-edited or manually reordered. The submitted 100 come from a fully deterministic ordering with explicit tie-breakers, so the ordering is fully determined by the seed within a fixed software environment.

Human intervention was confined to the code. Eight corrections were applied before the reported run: four from internal audit and four from external technical review.

Because hypervolume contribution feeds back into generator selection, these corrections required a fresh generation run rather than a rescore of the existing library. A rescore would reorder what already exists but cannot recover candidates the previous scoring rule prevented from being generated.

---

## Scope note: what this repository runs

The method described above is the full research pipeline: six generator arms, a
Thompson bandit over hypervolume contribution, APEX and HMD-AMP for activity,
HemoPI2 for haemolysis, and ESM3-open for secondary structure. That is the work.

The code in this repository is a self-contained distillation of it, and the
difference is deliberate rather than cosmetic. Three of those components cannot
be redistributed: APEX and HMD-AMP are third-party weights of 0.9 and 1.3 GB
under their own terms, and the ESM3 call needs a personal API credential. A
fourth, the ESM3 helix prediction, is a sampled structure prediction that returns
a different value for the same peptide on a later call, which a submission
required to reproduce byte for byte cannot contain.

So `uv run generate` samples from the quantum circuit Born machine prior over the
peptide latent space, applies the same compliance gate documented in section 3,
and ranks with the terms that can be recomputed from what is committed here plus
a DBAASP MIC prior. Everything it does is reproducible by anyone who clones this
repository; everything it omits is named above.

Figures quoted in the abstract, including the median APEX MIC of 4.33 uM and the
0.688 maximum pairwise identity, come from the research run (run A, seed
20260929). They describe that run, not the output of this entry point.
