"""Quantum VQVAE for antimicrobial peptide generation.

Architecture
------------
Sequence → ESMC-300M (frozen, d_model=960) → mean-pool [960]
  → Linear(960 → n_qubits) [pre-encoder]
  → AngleEmbedding + StronglyEntanglingLayers VQC [quantum bottleneck]
  → expval(PauliZ)^n_qubits ∈ [-1,1]^n
  → Linear(n_qubits → 960) [post-decoder]
  → Linear(960 → 64)  [token logits, ESMC vocab=64]

Training objective
------------------
L = CE(logits[1:-1], orig_tokens[1:-1]) + β × KL

where KL = 0.5 · Σ z_i² treats each Pauli-Z expectation as the mean of
N(z_i, 1) regularised toward the |+⟩^n prior (N(0,1) per qubit, which gives
⟨Z⟩=0 everywhere).

Gradient flow
-------------
The VQC node uses diff_method="parameter-shift" (exact analytic gradients
for gates parameterised by rotation angles, no finite-difference noise).
The ESM backbone is frozen, so gradients flow only through pre-encoder,
VQC weights, and post-decoder layers.

Fallback
--------
When ESMC/PennyLane are not installed the class degrades gracefully to a
pure-PyTorch dummy that passes the shape contracts (useful for unit testing
without GPU / big model weights).
"""
from __future__ import annotations

import logging
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

_ESMC_DIM = 960   # ESMC-300M hidden dimension
_ESMC_VOCAB = 64  # ESMC output vocab size

# Max sequence length supported by the learned positional embedding.
# Training data: 8-50 aa interior + BOS/EOS = up to 52 tokens.  64 gives margin.
_MAX_POS = 64

# Standard 20-AA alphabet for fallback decode.
# Kept as a local literal because peptide_gen is an upstream package that
# must not import from amp_challenger (would invert the dependency direction).
# When updating, mirror compliance.VALID_AMP_ALPHABET_STR in amp_challenger.
_AA_VOCAB = list("ACDEFGHIKLMNPQRSTVWY")

# ESMC token IDs for the 20 standard amino acids (contiguous range 4–23).
# Token 32 is <mask>, 0 is <cls>, 2 is <eos> — none of these are valid AA.
# decode() restricts argmax to this set so untrained models still return
# amino acid strings rather than special tokens.
_AA_TOKEN_IDS = list(range(4, 24))  # [4, 5, ..., 23]


def _token_ids_for(letters: str) -> list[int]:
    """Token ids that decode to any of ``letters``.

    Mirrors the id-to-letter rule used when no tokenizer is present
    (``_AA_VOCAB[i % len(_AA_VOCAB)]``), so a mask built here always matches what
    the decoder would actually emit.
    """
    wanted = set(letters.upper())
    return [
        t for t in _AA_TOKEN_IDS
        if _AA_VOCAB[t % len(_AA_VOCAB)] in wanted
    ]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sinusoidal_positional_encoding(
    seq_len: int, dim: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Standard transformer sinusoidal PE [seq_len, dim] (Vaswani 2017).

    Without this, the decoder broadcasts one ``h`` vector across all positions
    and the model collapses to poly-X output regardless of latent z.
    """
    pe = torch.zeros(seq_len, dim, device=device, dtype=dtype)
    position = torch.arange(seq_len, device=device, dtype=dtype).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, dim, 2, device=device, dtype=dtype)
        * (-math.log(10000.0) / dim)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
    return pe

def _mean_pool_embeddings(embeddings: torch.Tensor) -> torch.Tensor:
    """Mean-pool [1, L, D] → [D], excluding BOS (0) and EOS (-1) positions."""
    # embeddings: [1, L, D]  (batch=1 from a single-sequence forward)
    return embeddings[0, 1:-1].mean(dim=0)  # [D]


_VALID_AA_SET = frozenset(_AA_VOCAB)


def _tokens_to_sequence(token_ids: list[int], tokenizer: Any) -> str:
    """Decode a list of interior token IDs (no BOS/EOS) to a sequence string."""
    raw = tokenizer.decode(token_ids)
    # Keep only uppercase amino acid characters; drop spaces and special tokens.
    return "".join(c for c in raw.upper() if c in _VALID_AA_SET)


class _BinarizeSTE(torch.autograd.Function):
    """sign() forward, straight-through (hardtanh-clipped) gradient backward.

    Maps real pre-activations to a ``{-1, +1}`` latent code while keeping a usable
    gradient for the encoder. Inputs with ``|x| > 1`` receive zero gradient — the
    standard binary-net straight-through estimator (Hubara et al. 2016) — so the
    pre-encoder is discouraged from runaway magnitudes. This binary code is what
    the QCBM prior (Program II) learns to model: the circuit's measured bitstring
    *is* the latent.
    """

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        ctx.save_for_backward(x)
        return torch.where(x >= 0, torch.ones_like(x), -torch.ones_like(x))

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        (x,) = ctx.saved_tensors
        return grad_output * (x.abs() <= 1.0).to(grad_output.dtype)


def _binarize_ste(x: torch.Tensor) -> torch.Tensor:
    """Straight-through binarization to ``{-1, +1}`` (see :class:`_BinarizeSTE`)."""
    return _BinarizeSTE.apply(x)  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# WaveNet-style autoregressive decoder (ProtWave-VAE — PMC10911954, 2024)
# ---------------------------------------------------------------------------

class WaveNetDecoder(nn.Module):
    """Dilated causal convolutional decoder with per-position z conditioning.

    Each output token depends on (z, x_{<i}) — receptive field grows
    exponentially with depth.  For n_layers=4 the field covers up to 16
    positions; for n_layers=5, up to 32.  Sequence length cap is enforced
    by the caller (MAX_POS in the parent module).

    Architecture:
        token_embed(x_{<i}) + z_proj(z) → stack of gated dilated convs
        → output_head → token logits at every position

    Parameters
    ----------
    z_dim:
        Latent dimension (n_qubits).
    vocab_size:
        Output token vocabulary size (ESMC vocab = 64).
    hidden:
        Convolutional channel width.
    n_layers:
        Number of dilated causal conv layers.  Dilations are 1, 2, 4, ...
    """

    def __init__(
        self,
        z_dim: int,
        vocab_size: int,
        hidden: int = 128,
        n_layers: int = 4,
        dropout: float = 0.0,
        use_length_cond: bool = False,
        max_len: int = 64,
    ) -> None:
        super().__init__()
        self.hidden = hidden
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.dropout_p = dropout
        self.use_length_cond = use_length_cond

        self.token_embed = nn.Embedding(vocab_size, hidden)
        self.z_proj = nn.Linear(z_dim, hidden)
        # v9: optional length conditioning (explicit target-length embedding).
        if use_length_cond:
            self.length_embed = nn.Embedding(max_len, hidden)
        self.convs = nn.ModuleList([
            nn.Conv1d(hidden, hidden * 2, kernel_size=3, dilation=2 ** i)
            for i in range(n_layers)
        ])
        # v9: dropout between conv layers for regularisation.
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.out = nn.Linear(hidden, vocab_size)

    def _causal_conv(self, h: torch.Tensor, conv: nn.Conv1d, dilation: int) -> torch.Tensor:
        """Apply a causal 1d conv: pad left, ensure output[i] depends only on input[≤i]."""
        pad = (conv.kernel_size[0] - 1) * dilation
        h_padded = F.pad(h, (pad, 0))
        return conv(h_padded)

    def forward(
        self,
        z: torch.Tensor,
        tokens: torch.Tensor,
        target_len: int | None = None,
    ) -> torch.Tensor:
        """Autoregressive forward.

        Parameters
        ----------
        z:
            Latent vector, shape [z_dim].
        tokens:
            Input token sequence (teacher-forced during training), shape [L].
            Each output logit at position i conditions on tokens[<i] and z.
        target_len:
            v9 — optional explicit target length for length conditioning.
            When ``use_length_cond=True`` and ``target_len`` is supplied, an
            embedding for ``target_len`` is added to the input. Helps the
            decoder respect length constraints (typical AMP design need).

        Returns
        -------
        torch.Tensor
            Logits, shape [L, vocab_size].
        """
        L = tokens.shape[0]
        h = self.token_embed(tokens)                  # [L, hidden]
        h_z = self.z_proj(z).unsqueeze(0).expand(L, -1)  # [L, hidden]
        h = h + h_z
        if self.use_length_cond and target_len is not None:
            tl = max(0, min(int(target_len), self.length_embed.num_embeddings - 1))
            h_len = self.length_embed(torch.tensor(tl, device=z.device)).unsqueeze(0).expand(L, -1)
            h = h + h_len
        h = h.unsqueeze(0).transpose(1, 2)            # [1, hidden, L]

        for i, conv in enumerate(self.convs):
            gated = self._causal_conv(h, conv, dilation=2 ** i)  # [1, 2*hidden, L]
            tanh_part, sig_part = gated.chunk(2, dim=1)
            h_residual = torch.tanh(tanh_part) * torch.sigmoid(sig_part)
            h_residual = self.dropout(h_residual)        # v9: regularization
            h = h + h_residual

        h = h.transpose(1, 2).squeeze(0)              # [L, hidden]
        return self.out(h)                            # [L, vocab_size]


# ---------------------------------------------------------------------------
# MMD loss (ProtWave-VAE InfoMax-style)
# ---------------------------------------------------------------------------

def _gaussian_kernel(x: torch.Tensor, y: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """Pairwise Gaussian kernel matrix exp(-||x-y||²/(2σ²)).

    Parameters
    ----------
    x, y:
        Sample tensors of shape [n, d] and [m, d].
    sigma:
        Kernel bandwidth.

    Returns
    -------
    torch.Tensor
        Kernel matrix of shape [n, m].
    """
    x_sq = (x ** 2).sum(dim=1, keepdim=True)          # [n, 1]
    y_sq = (y ** 2).sum(dim=1, keepdim=True)          # [m, 1]
    sq_dists = x_sq + y_sq.t() - 2.0 * (x @ y.t())    # [n, m]
    return torch.exp(-sq_dists / (2.0 * sigma ** 2))


def mmd_loss(z_q: torch.Tensor, z_p: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """Unbiased Maximum Mean Discrepancy² between q(z) and p(z).

    Used by ProtWave-VAE (Information-Maximizing VAE) to prevent posterior
    collapse by penalising the aggregate posterior rather than per-sample
    KL.  Reference: arXiv:1706.02262, PMC10911954.

    Parameters
    ----------
    z_q:
        Encoder-sample batch from q(z|x) aggregated across the mini-batch,
        shape [B, d].
    z_p:
        Prior samples ~ p(z) = N(0, I), shape [B, d].
    sigma:
        Gaussian kernel bandwidth.

    Returns
    -------
    torch.Tensor
        Scalar MMD² loss.
    """
    k_qq = _gaussian_kernel(z_q, z_q, sigma).mean()
    k_pp = _gaussian_kernel(z_p, z_p, sigma).mean()
    k_qp = _gaussian_kernel(z_q, z_p, sigma).mean()
    return k_qq + k_pp - 2.0 * k_qp


# ---------------------------------------------------------------------------
# v11 — Contrastive (margin-based) energy-style training
# ---------------------------------------------------------------------------

def contrastive_margin_loss(
    pos_logp: torch.Tensor,
    neg_logp: torch.Tensor,
    margin: float = 1.0,
) -> torch.Tensor:
    """Pairwise margin loss: push pos > neg + margin in log-likelihood.

    Implements the energy-based training spirit by treating ``-log p(x|z)``
    as the model's energy function: real sequences should have low energy
    (high log-prob), random/noise sequences should have high energy
    (low log-prob), separated by at least ``margin`` nats per token.

    Reference: noise contrastive estimation (Gutmann & Hyvärinen 2010),
    margin ranking losses, energy-based GANs.

    Parameters
    ----------
    pos_logp:
        Per-sequence log-probability for the real (positive) sequence
        conditioned on z, shape [B] or scalar.
    neg_logp:
        Per-sequence log-probability for K random negatives conditioned
        on the same z, shape [B, K] or [K].
    margin:
        Minimum desired gap in log-prob between positive and negative.

    Returns
    -------
    torch.Tensor
        Scalar margin loss = ReLU(neg_logp − pos_logp + margin).
    """
    if pos_logp.ndim == 0:
        pos_logp = pos_logp.unsqueeze(0)
    if neg_logp.ndim == 1:
        neg_logp = neg_logp.unsqueeze(0)
    diff = neg_logp - pos_logp.unsqueeze(-1) + margin
    return torch.relu(diff).mean()


# ---------------------------------------------------------------------------
# QuantumVQVAE
# ---------------------------------------------------------------------------

class QuantumVQVAE(nn.Module):
    """Variational autoencoder with a quantum circuit bottleneck.

    Parameters
    ----------
    n_qubits:
        Width of the quantum latent space (and number of Pauli-Z measurements).
    n_layers:
        Depth of the ``StronglyEntanglingLayers`` block in the VQC.
    beta:
        Weight of the KL regularisation term.
    device_name:
        PennyLane device string.  ``"lightning.qubit"`` is fastest on CPU when
        ``pennylane-lightning`` is installed; falls back to ``"default.qubit"``.
    esm3_frozen:
        Freeze ESMC backbone weights (recommended — only train the classical
        projection layers + VQC weights).
    seed:
        RNG seed for the PennyLane device and VQC weight initialisation.
    """

    def __init__(
        self,
        n_qubits: int = 8,
        n_layers: int = 3,
        beta: float = 0.1,
        device_name: str = "lightning.qubit",
        esm3_frozen: bool = True,
        seed: int = 1337,
        esm_model: str = "esmc_300m",
        decoder_type: str = "broadcast",
        wavenet_hidden: int = 128,
        wavenet_layers: int = 4,
        wavenet_dropout: float = 0.0,
        use_length_cond: bool = False,
        use_quantum: bool = True,
        binary_latent: bool = False,
    ) -> None:
        super().__init__()
        # W4 ablation control: when False, the VQC is replaced by a learned
        # classical bottleneck of IDENTICAL width (n_qubits -> n_qubits Linear +
        # tanh) even if PennyLane is installed. Everything else (ESM backbone,
        # pre/post projections, decoder) is unchanged, so a classical-vs-quantum
        # A/B isolates exactly the bottleneck's contribution.
        self.use_quantum = use_quantum
        # Program II: when True the bottleneck is a parameter-free straight-through
        # binarizer (pre_encoder -> sign -> {-1,+1}^n_qubits). The QCBM prior models
        # p(b) over these codes, making the quantum circuit the *generative* model
        # (vs. the encoder-bottleneck VQC, which W4 showed adds nothing). Mutually
        # exclusive with the VQC / classical-linear bottlenecks below.
        self.binary_latent = binary_latent
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.beta = beta
        self.seed = seed
        # Backbone variant: "esmc_300m" (960-dim), "esmc_600m" (1152-dim).
        self.esm_model = esm_model
        # Decoder variant: "broadcast" (current) or "wavenet" (ProtWave-VAE style).
        self.decoder_type = decoder_type
        self.wavenet_hidden = wavenet_hidden
        self.wavenet_layers = wavenet_layers
        # v9: regularisation knobs on the WaveNet decoder.
        self.wavenet_dropout = wavenet_dropout
        self.use_length_cond = use_length_cond
        # v10: when True, skip the VQC inside _run_quantum() and use a
        # tanh(angles) classical bottleneck instead. The training script
        # toggles this off after the AE warm-up phase.
        self.ae_warmup_active: bool = False

        # ------------------------------------------------------------------
        # ESM backbone
        # ------------------------------------------------------------------
        self._esm: Any = None
        self._tokenizer: Any = None
        self._esm_dim: int = _ESMC_DIM
        self._esm_available: bool = False
        self._device: torch.device = torch.device("cpu")
        self._init_esm(esm3_frozen)

        # ------------------------------------------------------------------
        # Classical projection layers  (moved to same device as ESMC)
        # ------------------------------------------------------------------
        self.pre_encoder = nn.Linear(self._esm_dim, n_qubits).to(self._device)

        if self.decoder_type == "wavenet":
            # v3 — WaveNet decoder (ProtWave-VAE style).
            # No broadcast / pos_embed / LayerNorm — the WaveNet handles
            # positional structure via dilated causal convs internally.
            self.wavenet = WaveNetDecoder(
                z_dim=n_qubits,
                vocab_size=_ESMC_VOCAB,
                hidden=self.wavenet_hidden,
                n_layers=self.wavenet_layers,
                dropout=self.wavenet_dropout,
                use_length_cond=self.use_length_cond,
                max_len=_MAX_POS,
            ).to(self._device)
            # Keep dummy attributes so save/load works uniformly.
            self.post_decoder = nn.Linear(n_qubits, 1).to(self._device)
            self.pos_embed = nn.Embedding(1, 1).to(self._device)
            self.h_norm = nn.LayerNorm(1).to(self._device)
            self.output_head = nn.Linear(1, _ESMC_VOCAB).to(self._device)
        else:
            self.post_decoder = nn.Linear(n_qubits, self._esm_dim).to(self._device)
            self.pos_embed = nn.Embedding(_MAX_POS, self._esm_dim).to(self._device)
            self.h_norm = nn.LayerNorm(self._esm_dim).to(self._device)
            self.output_head = nn.Linear(self._esm_dim, _ESMC_VOCAB).to(self._device)
            self.wavenet = None

        # ------------------------------------------------------------------
        # Quantum circuit
        # ------------------------------------------------------------------
        self._qml: Any = None  # pennylane module ref
        self._qnode: Any = None
        self._q_weights: nn.Parameter | None = None
        self._quantum_available: bool = False
        if binary_latent:
            # Program II: parameter-free straight-through binary bottleneck — no
            # VQC and no classical linear layer. _run_quantum() short-circuits to
            # _binarize_ste(angles).
            logger.info(
                "QuantumVQVAE: binary_latent=True — straight-through %d-bit code "
                "bottleneck (QCBM-prior foundation)", self.n_qubits
            )
        elif use_quantum:
            self._init_quantum(device_name, seed)
        else:
            # W4 classical ablation arm: equal-width learned bottleneck, no VQC.
            self.classical_bottleneck = nn.Linear(self.n_qubits, self.n_qubits).to(self._device)
            logger.info(
                "QuantumVQVAE: use_quantum=False — classical bottleneck "
                "(W4 ablation, n_qubits=%d)", self.n_qubits
            )

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _init_esm(self, frozen: bool) -> None:
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._device = device

            from esm.models.esmc import ESMC
            from esm.utils.constants.models import ESMC_300M, ESMC_600M

            name = self.esm_model.lower()
            if name in {"esmc_600m", "esmc-600m", "600m"}:
                model = ESMC.from_pretrained(ESMC_600M, device=device)
                label = "ESMC-600M"
            else:
                model = ESMC.from_pretrained(ESMC_300M, device=device)
                label = "ESMC-300M"

            if frozen:
                for p in model.parameters():
                    p.requires_grad_(False)
                model.eval()

            # Auto-detect hidden dim from the loaded model's embedding
            # (works for ESMC.embed and ESM3.embed which are both nn.Embedding).
            try:
                detected_dim = int(model.embed.embedding_dim)
            except Exception:
                detected_dim = _ESMC_DIM   # fallback
            self._esm_dim = detected_dim

            self._esm = model
            self._tokenizer = model.tokenizer
            self._esm_available = True
            logger.info(
                "QuantumVQVAE: %s loaded (frozen=%s, device=%s, d_model=%d)",
                label, frozen, device, detected_dim,
            )
        except Exception as exc:
            logger.warning(
                "QuantumVQVAE: ESM unavailable (%s); using embedding fallback (dim=128)", exc
            )
            self._esm_dim = 128
            # Lightweight learnable fallback embedding over the 20-AA alphabet
            self.fallback_embed = nn.Embedding(len(_AA_VOCAB), 128)

    def _init_quantum(self, device_name: str, seed: int) -> None:
        try:
            import pennylane as qml
            self._qml = qml

            # Prefer lightning.qubit for speed; fall back to default.qubit
            try:
                dev = qml.device(device_name, wires=self.n_qubits, seed=seed)
            except Exception:
                logger.info(
                    "QuantumVQVAE: %s unavailable — using default.qubit", device_name
                )
                dev = qml.device("default.qubit", wires=self.n_qubits, seed=seed)

            n_qubits = self.n_qubits  # capture for closure

            @qml.qnode(dev, interface="torch", diff_method="parameter-shift")
            def circuit(inputs: torch.Tensor, weights: torch.Tensor) -> list:
                # Start in |+⟩^n — each qubit is in an equal superposition
                for i in range(n_qubits):
                    qml.Hadamard(wires=i)
                # Encode classical angles into rotation gates
                qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
                # Trainable strongly-entangling block
                qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
                return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

            self._qnode = circuit

            # VQC weight tensor — shape required by StronglyEntanglingLayers
            weight_shape = qml.StronglyEntanglingLayers.shape(
                n_layers=self.n_layers, n_wires=self.n_qubits
            )
            torch.manual_seed(seed)
            self._q_weights = nn.Parameter(torch.randn(weight_shape) * 0.01)
            self._quantum_available = True
            logger.info(
                "QuantumVQVAE: PennyLane circuit ready (n_qubits=%d, n_layers=%d, diff_method=parameter-shift)",
                self.n_qubits,
                self.n_layers,
            )
        except Exception as exc:
            logger.warning(
                "QuantumVQVAE: PennyLane unavailable (%s); using linear bottleneck fallback", exc
            )
            # Pure-classical fallback: a single linear layer acting as bottleneck
            self.classical_bottleneck = nn.Linear(self.n_qubits, self.n_qubits)

    # ------------------------------------------------------------------
    # Internal encoding helpers
    # ------------------------------------------------------------------

    def _embed_sequence(self, sequence: str) -> torch.Tensor:
        """Sequence string → mean-pooled embedding [esm_dim], no grad from ESM."""
        seq = sequence.upper().strip()

        if self._esm_available:
            from esm.sdk.api import ESMProtein

            protein = ESMProtein(sequence=seq)
            protein_tensor = self._esm.encode(protein)
            tokens = protein_tensor.sequence.unsqueeze(0)  # [1, L]
            with torch.no_grad():
                out = self._esm.forward(sequence_tokens=tokens)
            # ESMC-300M emits BFloat16; downstream Linear layers are Float32.
            return _mean_pool_embeddings(out.embeddings).to(torch.float32)  # [960]

        # Fallback: character-level embedding
        aa_to_idx = {aa: i for i, aa in enumerate(_AA_VOCAB)}
        indices = torch.tensor(
            [aa_to_idx.get(aa, 0) for aa in seq], dtype=torch.long, device=self._device
        )
        return self.fallback_embed(indices).mean(dim=0)  # [128]

    def _run_quantum(self, angles: torch.Tensor) -> torch.Tensor:
        """angles [n_qubits] → z [n_qubits] via VQC or classical fallback.

        When :attr:`ae_warmup_active` is True (v10 — Mentzer 2024 two-stage),
        the VQC is bypassed and a pure tanh classical bottleneck is used.
        This lets the encoder/decoder learn meaningful representations before
        the parameter-shift VQC constrains them.
        """
        if self.binary_latent:
            # Program II: straight-through {-1,+1}^n code (range matches the
            # PauliZ/tanh latent the decoder already expects).
            return _binarize_ste(angles)
        if self.ae_warmup_active:
            # v10: skip VQC during AE warm-up; use plain tanh.
            return torch.tanh(angles)
        if self._quantum_available:
            result = self._qnode(angles, self._q_weights)
            # PennyLane torch interface may return a list or stacked tensor
            if isinstance(result, (list, tuple)):
                return torch.stack(result)
            return result
        # Classical bottleneck (tanh keeps values in [-1, 1] like PauliZ expval)
        return torch.tanh(self.classical_bottleneck(angles))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode(self, sequence: str) -> torch.Tensor:
        """Sequence string → quantum latent vector z ∈ [-1, 1]^n_qubits.

        Gradients flow through ``pre_encoder`` and the VQC weights via the
        parameter-shift rule.  The ESM embeddings are detached (frozen).
        """
        emb = self._embed_sequence(sequence)            # [esm_dim]
        angles = self.pre_encoder(emb)                  # [n_qubits]
        z = self._run_quantum(angles)                   # [n_qubits]
        return z

    def decode(
        self,
        z: torch.Tensor,
        seq_len: int | None = None,
        temperature: float = 0.0,
        top_k: int = 0,
        repetition_penalty: float = 1.0,
        exclude_aas: str = "",
    ) -> str:
        """Quantum latent vector z → amino acid sequence string.

        Parameters
        ----------
        z:
            Latent vector of shape ``[n_qubits]``.
        seq_len:
            Target sequence length (number of residues, excluding BOS/EOS).
            When ``None``, uses ``n_qubits`` as a heuristic default.
        temperature:
            Sampling temperature. ``0.0`` → greedy argmax (deterministic).
            ``> 0`` → softmax sampling, larger values increase diversity.
            Typical: 0.7 for balanced diversity, 1.0 for max stochasticity.
            Only affects the WaveNet path; broadcast decode is always greedy.
        top_k:
            If > 0, restrict sampling to the top-k highest-probability AA tokens
            at each step (after temperature scaling). 0 = no restriction.
            Typical: 5-10 for AMP generation. Only applies when temperature > 0.
        repetition_penalty:
            Multiplier > 1.0 reduces probability of tokens already produced in
            the current sequence (suppresses poly-X patterns).
            1.0 = no penalty. 1.2-1.5 = moderate. Reference: CTRL paper.
        """
        if seq_len is None:
            seq_len = self.n_qubits

        # encode() returns a CPU tensor while the WaveNet weights live on the
        # model device, so a latent handed straight from encode() to decode()
        # raised "Expected all tensors to be on the same device" on every call.
        # decode_bits() already moved it; decode() did not, and that asymmetry
        # is why the QCBM-prior path worked and the VQVAE arm silently produced
        # nothing in every run to date.
        z = z.to(self._device)

        if self.decoder_type == "wavenet" and self.wavenet is not None:
            # v3 — autoregressive WaveNet decode (token-by-token).
            # Start with BOS (token 0) and pick from AA token set only.
            aa_bias = torch.full(
                (_ESMC_VOCAB,), float("-inf"), device=z.device, dtype=z.dtype
            )
            aa_bias[torch.tensor(_AA_TOKEN_IDS, dtype=torch.long, device=z.device)] = 0.0
            # Residues the caller's design space forbids never get sampled,
            # instead of being generated and discarded downstream.
            if exclude_aas:
                _drop = _token_ids_for(exclude_aas)
                if _drop:
                    aa_bias[torch.tensor(_drop, dtype=torch.long, device=z.device)] = float("-inf")
            tokens = [0]   # BOS
            # v9 — pass target length when length conditioning is active so
            # decode respects the requested seq_len more faithfully.
            tgt_len = seq_len if self.use_length_cond else None
            for _ in range(min(seq_len, _MAX_POS) - 1):
                inp = torch.tensor(tokens, dtype=torch.long, device=z.device)
                logits_step = self.wavenet(z, inp, target_len=tgt_len)  # [len(tokens), vocab]
                logits_aa = logits_step[-1] + aa_bias

                # Repetition penalty: down-weight tokens already generated.
                # CTRL-style: logit /= penalty if logit > 0 else logit *= penalty.
                if repetition_penalty != 1.0 and len(tokens) > 1:
                    for t in set(tokens[1:]):   # skip BOS
                        if logits_aa[t] > 0:
                            logits_aa[t] = logits_aa[t] / repetition_penalty
                        else:
                            logits_aa[t] = logits_aa[t] * repetition_penalty

                if temperature > 0.0:
                    # Top-k truncation: keep only top_k highest-logit tokens.
                    if top_k > 0:
                        topk_vals, topk_idx = torch.topk(logits_aa, k=min(top_k, logits_aa.shape[0]))
                        mask = torch.full_like(logits_aa, float("-inf"))
                        mask[topk_idx] = topk_vals
                        logits_aa = mask
                    probs = torch.softmax(logits_aa / temperature, dim=-1)
                    next_tok = int(torch.multinomial(probs, num_samples=1).item())
                else:
                    next_tok = int(logits_aa.argmax().item())
                tokens.append(next_tok)
            token_ids = tokens[1:]   # drop BOS
            if self._tokenizer is not None:
                return _tokens_to_sequence(token_ids, self._tokenizer)
            return "".join(_AA_VOCAB[i % len(_AA_VOCAB)] for i in token_ids)

        h = self.post_decoder(z)                        # [esm_dim]
        # Clamp position indices to the learned embedding range.
        pos_ids = torch.arange(min(seq_len, _MAX_POS), device=h.device)
        pe = self.pos_embed(pos_ids)                    # [seq_len, esm_dim]
        h_expanded = self.h_norm(
            h.unsqueeze(0).expand(pe.shape[0], -1) + pe
        )                                                # [seq_len, esm_dim]
        logits = self.output_head(h_expanded)            # [seq_len, vocab=64]

        # Restrict argmax to valid AA token IDs so the untrained model
        # never decodes to <mask> or other special tokens.
        aa_bias = torch.full(
            (logits.shape[-1],), float("-inf"), device=logits.device, dtype=logits.dtype
        )
        aa_bias[torch.tensor(_AA_TOKEN_IDS, dtype=torch.long)] = 0.0
        token_ids = (logits + aa_bias).argmax(dim=-1).tolist()  # [seq_len]

        if self._tokenizer is not None:
            return _tokens_to_sequence(token_ids, self._tokenizer)

        # Fallback: map IDs mod 20 → _AA_VOCAB
        return "".join(_AA_VOCAB[i % len(_AA_VOCAB)] for i in token_ids)

    def forward(self, sequence: str) -> dict[str, torch.Tensor]:
        """Full encode–decode pass; returns loss components.

        Returns
        -------
        dict with keys:
            ``z``           — quantum latent [n_qubits]
            ``recon_loss``  — token-level cross-entropy (scalar)
            ``kl_loss``     — KL proxy: 0.5 · Σ z_i² (scalar)
            ``total_loss``  — recon_loss + β · kl_loss (scalar)
        """
        seq = sequence.upper().strip()
        emb = self._embed_sequence(seq)           # [esm_dim], detached from ESM
        angles = self.pre_encoder(emb)            # [n_qubits]
        z = self._run_quantum(angles)             # [n_qubits]

        # KL: treat z_i as mean of N(z_i, 1) vs N(0,1) prior.
        # Binary latent: z_i ∈ {-1,+1} so 0.5·Σz² is a constant (no gradient);
        # the prior over codes is learned separately by the QCBM, not regularised
        # here, so the KL term is dropped.
        if self.binary_latent:
            kl_loss = torch.zeros((), device=z.device, dtype=z.dtype)
        else:
            kl_loss = 0.5 * (z ** 2).sum()

        # Decode
        h = self.post_decoder(z)                  # [esm_dim]

        # Get original token sequence for reconstruction target
        if self._esm_available:
            from esm.sdk.api import ESMProtein

            protein = ESMProtein(sequence=seq)
            protein_tensor = self._esm.encode(protein)
            orig_tokens = protein_tensor.sequence   # [L], includes BOS/EOS
        else:
            # Fallback: dummy token IDs [0, 1, ..., L-1] (test-only)
            L = len(seq) + 2
            orig_tokens = torch.arange(L, dtype=torch.long, device=self._device) % _ESMC_VOCAB

        L = orig_tokens.shape[0]
        L_eff = min(L, _MAX_POS)
        orig_eff = orig_tokens[:L_eff]

        if self.decoder_type == "wavenet" and self.wavenet is not None:
            # v3 — WaveNet teacher-forcing: feed shifted token sequence; at
            # position i the model sees orig_eff[:i] and must predict orig_eff[i].
            # The convs are causal, so the loss at position i only sees x_{<i}.
            # v9 — when length conditioning is on, pass interior residue count
            # (L_eff - 2 for BOS/EOS) so the decoder knows the target length.
            tgt_len = max(0, L_eff - 2) if self.use_length_cond else None
            logits = self.wavenet(z, orig_eff, target_len=tgt_len)  # [L_eff, vocab]
            # Predict positions 1..L-1 from contexts ending at 0..L-2.
            # logits[i] is conditioned on orig_eff[:i+1] but we want it to
            # predict orig_eff[i+1] (next token prediction). Standard NTP shift:
            recon_loss = F.cross_entropy(logits[:-1], orig_eff[1:])
        else:
            pos_ids = torch.arange(L_eff, device=h.device)
            pe = self.pos_embed(pos_ids)                    # [L_eff, esm_dim]
            h_expanded = self.h_norm(
                h.unsqueeze(0).expand(L_eff, -1) + pe
            )                                                # [L_eff, esm_dim]
            logits = self.output_head(h_expanded)            # [L_eff, 64]

            # Reconstruction loss over residue positions (skip BOS=0 and EOS=-1)
            recon_loss = F.cross_entropy(logits[1:-1], orig_eff[1:-1])
        total_loss = recon_loss + self.beta * kl_loss

        return {
            "z": z,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
            "total_loss": total_loss,
        }

    # ------------------------------------------------------------------
    # Convenience generation
    # ------------------------------------------------------------------

    def score_seq(self, z: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        """Per-sequence total log-probability of ``tokens`` given ``z``.

        Used by v11 contrastive training to score positives/negatives. Only
        valid for ``decoder_type='wavenet'`` (autoregressive log-prob).

        Parameters
        ----------
        z:
            Latent vector, shape [n_qubits].
        tokens:
            Token sequence including BOS/EOS, shape [L].

        Returns
        -------
        torch.Tensor
            Scalar log-probability sum over predicted positions (1..L-1).
        """
        if self.decoder_type != "wavenet" or self.wavenet is None:
            return torch.tensor(0.0, device=z.device)
        L_eff = min(tokens.shape[0], _MAX_POS)
        tk = tokens[:L_eff]
        tgt_len = max(0, L_eff - 2) if self.use_length_cond else None
        logits = self.wavenet(z, tk, target_len=tgt_len)
        # Sum of log-prob of true tokens[1:] given context tokens[:-1].
        log_probs = F.log_softmax(logits[:-1], dim=-1)
        return log_probs.gather(1, tk[1:].unsqueeze(1)).sum()

    def generate(
        self,
        seq_len: int = 16,
        noise_scale: float = 1.0,
        seed: int | None = None,
        temperature: float = 0.0,
        top_k: int = 0,
        repetition_penalty: float = 1.0,
    ) -> str:
        """Sample a random latent point and decode to an amino acid sequence.

        The prior is |+⟩^n → ⟨Z⟩ = 0 per qubit, so we sample from N(0, 1)
        and clip to [-1, 1] to stay in the physical range of Pauli-Z.

        Parameters
        ----------
        seq_len:
            Desired sequence length (residues).
        noise_scale:
            Standard deviation of the Gaussian prior sample (default 1.0).
        seed:
            Optional RNG seed for reproducibility.
        temperature, top_k, repetition_penalty:
            Forwarded to :meth:`decode`. Defaults preserve greedy behaviour
            for backward compatibility.
        """
        rng = torch.Generator(device=self._device)
        if seed is not None:
            rng.manual_seed(seed)
        z = torch.randn(self.n_qubits, generator=rng, device=self._device) * noise_scale
        z = z.clamp(-1.0, 1.0)
        with torch.no_grad():
            return self.decode(
                z,
                seq_len=seq_len,
                temperature=temperature,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
            )

    # ------------------------------------------------------------------
    # Binary-latent API (Program II — QCBM prior coupling)
    # ------------------------------------------------------------------

    def encode_bits(self, sequence: str) -> torch.Tensor:
        """Sequence → binary latent code ``b ∈ {0,1}^n_qubits`` (long tensor).

        Only meaningful when ``binary_latent=True``: the straight-through sign
        bottleneck yields a ``±1`` code, mapped here to ``{0,1}``. The empirical
        distribution of these codes over the training corpus is the QCBM's target.
        """
        with torch.no_grad():
            z = self.encode(sequence)            # {-1,+1}^n in binary mode
        return (z > 0).to(torch.long)

    @staticmethod
    def _bits_to_latent(bits: torch.Tensor) -> torch.Tensor:
        """``{0,1}^n`` → ``{-1,+1}^n`` float latent the decoder expects."""
        return bits.to(torch.float32) * 2.0 - 1.0

    def decode_bits(
        self,
        bits: torch.Tensor,
        seq_len: int | None = None,
        temperature: float = 0.0,
        top_k: int = 0,
        repetition_penalty: float = 1.0,
        exclude_aas: str = "",
    ) -> str:
        """Binary code ``{0,1}^n`` → peptide via the trained decoder.

        This is the generation path for the QCBM-prior arm: a bitstring sampled
        from the QCBM is decoded here.
        """
        z = self._bits_to_latent(bits).to(self._device)
        with torch.no_grad():
            return self.decode(
                z,
                seq_len=seq_len,
                temperature=temperature,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                exclude_aas=exclude_aas,
            )

    @property
    def quantum_available(self) -> bool:
        return self._quantum_available

    @property
    def esm_available(self) -> bool:
        return self._esm_available
