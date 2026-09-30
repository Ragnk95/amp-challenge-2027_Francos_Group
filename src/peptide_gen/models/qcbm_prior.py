"""Quantum Circuit Born Machine (QCBM) prior for the AMP generator (Program II).

The QCBM is the *generative* quantum model: a parameterised circuit whose Born-rule
measurement distribution over ``n_qubits`` defines a learnable distribution over the
binary latent codes produced by the binary-latent ``QuantumVQVAE`` (Phase G0). At
generation time we sample a bitstring from the QCBM and decode it with the classical
WaveNet decoder — so quantum does the generative work, unlike the encoder-bottleneck
VQC that the W4 ablation found inert.

Why a Born machine on the *prior* (not the bottleneck)? A VAE's fixed Gaussian prior
is a known weakness (the "prior hole" / aggregated-posterior mismatch). A QCBM is the
canonical quantum generative distribution (Born rule over measured bitstrings; Liu &
Wang 2018; Benedetti et al. 2019), so it is placed exactly where the deficiency is.

Ansatz
------
Hardware-efficient, IBM-transpilable: per layer, an ``RY``+``RZ`` rotation on every
qubit followed by a ``CNOT`` ring (``q -> (q+1) mod n``). Parameters
``weights[n_layers, n_qubits, 2]``. The same ansatz is reconstructed in Qiskit by the
Phase-G3 hardware-sampling script using :meth:`get_weights` / :attr:`ansatz`.

Backend
-------
For ``n_qubits <= ~16`` the full ``2**n`` probability vector is enumerable, so we use
PennyLane ``default.qubit`` with ``diff_method="backprop"`` for *exact* differentiable
probabilities. Training then minimises the exact negative log-likelihood of the
empirical code histogram (the gold standard for small Born machines), with an optional
sample-based MMD reported alongside. PennyLane is imported lazily so this module (and
its bit-ordering helpers) import even where PennyLane is absent.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

ANSATZ = "ry_rz_cnot_ring"


# ---------------------------------------------------------------------------
# Bit-ordering helpers (shared with the trainer + hardware script so the
# empirical histogram, the QCBM probs, and the decoded codes all agree).
# Convention: qubit/latent dim 0 is the MOST significant bit, matching
# ``qml.probs(wires=range(n))`` and Qiskit's big-endian readout after reversal.
# ---------------------------------------------------------------------------


def bits_to_index(bits: np.ndarray) -> np.ndarray:
    """``[..., n]`` of {0,1} → integer index, qubit 0 = MSB."""
    bits = np.asarray(bits)
    n = bits.shape[-1]
    weights = (1 << np.arange(n - 1, -1, -1)).astype(np.int64)
    return (bits.astype(np.int64) * weights).sum(axis=-1)


def index_to_bits(idx: np.ndarray | int, n: int) -> np.ndarray:
    """Integer index → ``[..., n]`` of {0,1}, qubit 0 = MSB."""
    idx = np.asarray(idx, dtype=np.int64)
    shifts = np.arange(n - 1, -1, -1)
    return ((idx[..., None] >> shifts) & 1).astype(np.uint8)


def empirical_histogram(codes: np.ndarray, n_qubits: int) -> torch.Tensor:
    """``codes`` [N, n] of {0,1} → normalised probability vector ``[2**n]`` (float64)."""
    idx = bits_to_index(np.asarray(codes))
    hist = np.bincount(idx, minlength=1 << n_qubits).astype(np.float64)
    total = hist.sum()
    if total <= 0:
        raise ValueError("empirical_histogram: no codes")
    return torch.from_numpy(hist / total)


# ---------------------------------------------------------------------------
# QCBM
# ---------------------------------------------------------------------------


class QCBMPrior(nn.Module):
    """Quantum Circuit Born Machine over ``n_qubits`` (enumerable for n ≲ 16)."""

    def __init__(
        self,
        n_qubits: int,
        n_layers: int = 4,
        device_name: str = "default.qubit",
        seed: int = 1337,
        init_std: float = 0.1,
    ) -> None:
        super().__init__()
        if n_qubits > 20:
            raise ValueError(f"QCBMPrior enumerates 2**n states; n_qubits={n_qubits} too large")
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.ansatz = ANSATZ
        torch.manual_seed(seed)
        # weights[layer, qubit, {RY, RZ}]
        self.weights = nn.Parameter(torch.randn(n_layers, n_qubits, 2) * init_std)
        self._qnode: Any = None
        self._build_qnode(device_name, seed)

    # ------------------------------------------------------------------
    def _build_qnode(self, device_name: str, seed: int) -> None:
        import pennylane as qml  # lazy — keeps module importable without PennyLane

        try:
            dev = qml.device(device_name, wires=self.n_qubits, seed=seed)
        except Exception:  # noqa: BLE001
            dev = qml.device("default.qubit", wires=self.n_qubits, seed=seed)
        n_qubits, n_layers = self.n_qubits, self.n_layers

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(weights: torch.Tensor) -> Any:
            for layer in range(n_layers):
                for q in range(n_qubits):
                    qml.RY(weights[layer, q, 0], wires=q)
                    qml.RZ(weights[layer, q, 1], wires=q)
                for q in range(n_qubits):
                    qml.CNOT(wires=[q, (q + 1) % n_qubits])
            return qml.probs(wires=range(n_qubits))

        self._qnode = circuit
        logger.info(
            "QCBMPrior ready: n_qubits=%d n_layers=%d ansatz=%s device=%s",
            self.n_qubits, self.n_layers, self.ansatz, device_name,
        )

    # ------------------------------------------------------------------
    def probabilities(self) -> torch.Tensor:
        """Exact Born distribution ``p_θ(b)`` as a differentiable ``[2**n]`` tensor."""
        p = self._qnode(self.weights)
        if isinstance(p, (list, tuple)):
            p = torch.stack(list(p))
        return p.to(torch.float64)

    def nll(self, target_hist: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        """Negative log-likelihood of the empirical histogram under ``p_θ`` (scalar).

        Equivalent (up to a constant) to ``KL(q || p_θ)`` — the maximum-likelihood
        objective for the Born machine. Exact and differentiable for enumerable n.
        """
        p = self.probabilities()
        q = target_hist.to(p.dtype).to(p.device)
        return -(q * torch.log(p + eps)).sum()

    def entropy(self, eps: float = 1e-12) -> float:
        with torch.no_grad():
            p = self.probabilities()
            return float(-(p * torch.log(p + eps)).sum().item())

    @torch.no_grad()
    def sample(self, n_shots: int, seed: int | None = None) -> np.ndarray:
        """Draw ``n_shots`` bitstrings ``[n_shots, n_qubits]`` (uint8) from ``p_θ``.

        Simulator path: multinomial draw from the exact probability vector. The
        real-hardware path lives in ``scripts/run_qcbm_hardware_sample.py``.
        """
        p = self.probabilities().detach().cpu().numpy()
        p = np.clip(p, 0.0, None)
        p = p / p.sum()
        rng = np.random.default_rng(seed)
        idx = rng.choice(p.shape[0], size=int(n_shots), p=p)
        return index_to_bits(idx, self.n_qubits)

    # ------------------------------------------------------------------
    def get_weights(self) -> np.ndarray:
        """Trained angles ``[n_layers, n_qubits, 2]`` for Qiskit reconstruction (G3)."""
        return self.weights.detach().cpu().numpy()

    def config(self) -> dict[str, Any]:
        return {
            "n_qubits": self.n_qubits,
            "n_layers": self.n_layers,
            "ansatz": self.ansatz,
        }
