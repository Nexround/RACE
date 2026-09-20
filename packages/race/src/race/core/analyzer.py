"""Core RACE analysis algorithms.

RACE (Bayesian Residual Alignment for Consistency Estimation) quantifies
neuron functional consistency with Normal-Inverse-Gamma posterior inference.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from scipy.stats import t as scipy_t

try:
    import torch

    _TORCH_AVAILABLE = True
except ImportError as e:
    raise ImportError(
        "PyTorch is required for RACE analysis. "
        "Please install it with: uv pip install torch"
    ) from e


# Tensor utilities


def _to_tensor(
    x: torch.Tensor, *, device: Optional["torch.device"] = None
) -> "torch.Tensor":
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"x must be a torch.Tensor, got {type(x).__name__}")
    if device is not None and x.device != device:
        return x.to(device=device)
    return x


# Layer weights


@dataclass
class LayerWeights:
    """Weight matrices for a transformer layer.

    All weights are stored as torch.Tensor for efficient computation.
    """

    W_O: Optional[torch.Tensor] = (
        None  # [d_attn, d_model] - attention output projection
    )
    W_down: Optional[torch.Tensor] = None  # [d_model, K]      - FFN down projection


# Normal-Inverse-Gamma posterior


class NIGPosterior:
    """Online Normal-Inverse-Gamma posterior for neuron functional consistency.

    Models signed alignment scores  e_j ~ N(mu_j, sigma_j^2)  and jointly
    infers (mu_j, sigma_j^2) via conjugate NIG updating.

    Prior: (mu, sigma^2) ~ NIG(mu_0, lambda_0, alpha_0, beta_0)
    - mu_0    : prior mean (0 → no directional preference)
    - lambda_0: prior pseudo-count for mean
    - alpha_0 : prior shape for variance (> 0)
    - beta_0  : prior scale for variance (> 0)

    After observing N samples {e_t}:
        lambda_n = lambda_0 + N
        mu_n     = (lambda_0 * mu_0 + N * e_bar) / lambda_n
        alpha_n  = alpha_0 + N / 2
        beta_n   = beta_0 + S/2 + lambda_0*N*(e_bar - mu_0)^2 / (2*lambda_n)
    where e_bar = sample mean, S = sum of squared deviations.

    Functional consistency score: signed posterior mean mu_n.
    CAM score: I[mu_n > 0] max(0, mu_n - t * sigma_mu)
    """

    def __init__(
        self,
        dim: int,
        mu_0: float = 0.0,
        lambda_0: float = 1.0,
        alpha_0: float = 1.0,
        beta_0: float = 1.0,
        device: Optional[torch.device] = None,
    ) -> None:
        if device is None:
            device = torch.device("cpu")
        self.dim = dim
        self.device = device

        # Prior hyperparameters (scalars — shared across all neurons in this module/layer)
        self.mu_0 = mu_0
        self.lambda_0 = lambda_0
        self.alpha_0 = alpha_0
        self.beta_0 = beta_0

        # Sufficient statistics accumulators — per neuron, shape [dim]
        self._N: int = 0
        self._sum_e: torch.Tensor = torch.zeros(dim, dtype=torch.float32, device=device)
        self._sum_e2: torch.Tensor = torch.zeros(
            dim, dtype=torch.float32, device=device
        )

    # ------------------------------------------------------------------
    # Online update
    # ------------------------------------------------------------------

    def add_evidence(self, evidence: torch.Tensor) -> None:
        """Accumulate evidence from one or more observations.

        Args:
            evidence: Signed alignment scores, shape [dim] (single token/sample)
                      or [N, dim] (batch of tokens/samples). Values in (-inf, +inf).
        """
        if not isinstance(evidence, torch.Tensor):
            raise TypeError(
                f"evidence must be torch.Tensor, got {type(evidence).__name__}"
            )

        ev = evidence.detach().to(dtype=torch.float32, device=self.device)
        if ev.ndim == 1:
            if ev.shape[0] != self.dim:
                raise ValueError(
                    f"evidence dim {ev.shape[0]} != posterior dim {self.dim}"
                )
            self._N += 1
            self._sum_e.add_(ev)
            self._sum_e2.add_(ev * ev)
        elif ev.ndim == 2:
            if ev.shape[1] != self.dim:
                raise ValueError(
                    f"evidence dim {ev.shape[1]} != posterior dim {self.dim}"
                )
            n = ev.shape[0]
            self._N += n
            self._sum_e.add_(ev.sum(dim=0))
            self._sum_e2.add_((ev * ev).sum(dim=0))
        else:
            raise ValueError(f"evidence must be 1-D or 2-D, got {ev.ndim}-D")

    # ------------------------------------------------------------------
    # Posterior parameters (computed from sufficient statistics)
    # ------------------------------------------------------------------

    @property
    def N(self) -> int:
        return self._N

    @property
    def lambda_n(self) -> float:
        return self.lambda_0 + self._N

    @property
    def alpha_n(self) -> float:
        return self.alpha_0 + self._N / 2.0

    def _e_bar(self) -> torch.Tensor:
        """Per-neuron sample mean, shape [dim], float32."""
        if self._N == 0:
            return torch.zeros(self.dim, dtype=torch.float32, device=self.device)
        return self._sum_e / self._N

    def _S(self) -> torch.Tensor:
        """Per-neuron sum of squared deviations S = Σ(e_t - e_bar)^2."""
        if self._N == 0:
            return torch.zeros(self.dim, dtype=torch.float32, device=self.device)
        e_bar = self._e_bar()
        return self._sum_e2 - self._N * (e_bar * e_bar)

    @property
    def mu_n(self) -> torch.Tensor:
        """Posterior mean of mu (signed functional consistency), shape [dim]."""
        e_bar = self._e_bar()
        return (self.lambda_0 * self.mu_0 + self._N * e_bar) / self.lambda_n

    @property
    def beta_n(self) -> torch.Tensor:
        """Posterior scale of variance prior, shape [dim]."""
        e_bar = self._e_bar()
        S = self._S()
        discrepancy = (
            self.lambda_0 * self._N * (e_bar - self.mu_0) ** 2 / (2.0 * self.lambda_n)
        )
        return self.beta_0 + S / 2.0 + discrepancy

    @property
    def posterior_mean(self) -> torch.Tensor:
        """Signed posterior mean mu_n (functional consistency score), shape [dim].

        Positive → stable excitatory neuron; Negative → stable inhibitory neuron;
        Near-zero → no consistent contribution.
        """
        return self.mu_n.to(dtype=torch.float32)

    @property
    def empirical_mean(self) -> torch.Tensor:
        """Pure empirical mean of alignment scores: sum_e / N (no prior), shape [dim].

        Ablation baseline: direct arithmetic mean without Bayesian regularisation.
        """
        return self._e_bar().to(dtype=torch.float32)

    @property
    def empirical_snr(self) -> torch.Tensor:
        """Empirical SNR: max(0, mean(e)) / sample_std(e), shape [dim].

        Ablation baseline simulating a frequentist variance-penalised score.
        Returns zeros when N < 2 (undefined standard deviation).
        """
        if self._N < 2:
            return torch.zeros(self.dim, dtype=torch.float32, device=self.device)
        e_bar = self._e_bar()
        # Frequentist baseline: use the unbiased sample variance (N - 1),
        # matching the empirical standard-deviation definition in the paper.
        var = torch.clamp(self._S() / (self._N - 1), min=0.0)
        return (torch.clamp(e_bar, min=0.0) / (torch.sqrt(var) + 1e-8)).to(
            dtype=torch.float32
        )

    @property
    def posterior_variance_of_mean(self) -> torch.Tensor:
        """Posterior variance of mu: beta_n / (alpha_n * lambda_n), shape [dim].

        Reflects uncertainty in the mean estimate. Small value = high confidence.
        """
        return (self.beta_n / (self.alpha_n * self.lambda_n)).to(dtype=torch.float32)

    # ------------------------------------------------------------------
    # CAM score
    # ------------------------------------------------------------------

    def cam_score(self, gamma: float = 0.05) -> torch.Tensor:
        """Positive Conservative Alignment Magnitude (CAM) score.

        CAM_j(gamma) = I[mu_n,j > 0]
                       max(0, mu_n,j - t_{1-gamma, 2*alpha_n} sigma_{mu,j})

        where sigma_{mu,j} = sqrt(beta_n,j / (alpha_n * lambda_n))

        A nonzero CAM indicates statistically verified positive alignment.

        Args:
            gamma: One-tailed confidence level (default 0.05 → 95% CAM).

        Returns:
            cam: Non-negative tensor, shape [dim].
        """
        mu = self.mu_n  # [dim], float32
        bn = self.beta_n  # [dim], float32
        an = self.alpha_n  # scalar
        ln = self.lambda_n  # scalar

        sigma_mu = torch.sqrt(bn / (an * ln))  # [dim]

        # t-quantile with 2*alpha_n degrees of freedom
        df = 2.0 * an
        t_q = float(scipy_t.ppf(1.0 - gamma, df))

        cam = torch.where(
            mu > 0,
            torch.clamp(mu - t_q * sigma_mu, min=0.0),
            torch.zeros_like(mu),
        )
        return cam.to(dtype=torch.float32)

    def negative_cam_score(self, gamma: float = 0.05) -> torch.Tensor:
        """Negative-CAM ablation score from the paper.

        NegCAM_j(gamma) = I[mu_n,j < 0]
                          max(0, -mu_n,j - t * sigma_{mu,j})
        """
        mu = self.mu_n
        sigma_mu = torch.sqrt(self.beta_n / (self.alpha_n * self.lambda_n))
        t_q = float(scipy_t.ppf(1.0 - gamma, 2.0 * self.alpha_n))
        neg_cam = torch.where(
            mu < 0,
            torch.clamp(-mu - t_q * sigma_mu, min=0.0),
            torch.zeros_like(mu),
        )
        return neg_cam.to(dtype=torch.float32)

    # ------------------------------------------------------------------
    # State dict (for H5 serialization)
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        """Return serializable NIG state."""
        return {
            "dim": self.dim,
            "N": self._N,
            "sum_e": self._sum_e.cpu().numpy().astype("float32"),
            "sum_e2": self._sum_e2.cpu().numpy().astype("float32"),
            "mu_0": self.mu_0,
            "lambda_0": self.lambda_0,
            "alpha_0": self.alpha_0,
            "beta_0": self.beta_0,
        }

    @classmethod
    def from_state_dict(
        cls,
        state: dict,
        dim: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> "NIGPosterior":
        """Restore NIGPosterior from a state dict.

        ``dim`` may be omitted when it is already encoded in ``state["dim"]``
        (which is the case for dicts produced by :meth:`state_dict`).
        """
        import numpy as np

        if device is None:
            device = torch.device("cpu")
        resolved_dim = (
            dim
            if dim is not None
            else int(state.get("dim", len(np.asarray(state["sum_e"]))))
        )
        obj = cls(
            dim=resolved_dim,
            mu_0=float(state["mu_0"]),
            lambda_0=float(state["lambda_0"]),
            alpha_0=float(state["alpha_0"]),
            beta_0=float(state["beta_0"]),
            device=device,
        )
        obj._N = int(state["N"])
        obj._sum_e = torch.as_tensor(
            np.asarray(state["sum_e"], dtype=np.float32),
            dtype=torch.float32,
            device=device,
        )
        obj._sum_e2 = torch.as_tensor(
            np.asarray(state["sum_e2"], dtype=np.float32),
            dtype=torch.float32,
            device=device,
        )
        return obj


# RACE configuration and state


@dataclass
class RACEConfig:
    """Configuration for RACE analysis."""

    token_mode: str = "last_k"  # 'all' or 'last_k'
    last_k: int = 128
    # NIG prior hyperparameters
    mu_0: float = 0.0  # prior mean (no directional preference)
    lambda_0: float = 1.0  # prior pseudo-count for mean
    alpha_0: float = 1.0  # prior shape for variance
    beta_0: float = 1.0  # prior scale for variance
    gamma: float = 0.05  # CAM confidence level (1-gamma upper t-quantile)
    eps: float = 1e-8  # numerical stability


@dataclass
class RACEState:
    """Per-layer NIG posteriors for one semantic group."""

    nig_attn: Dict[int, NIGPosterior]
    nig_ffn: Dict[int, NIGPosterior]


def select_token_indices(num_tokens: int, mode: str, last_k: int) -> List[int]:
    if mode == "all" or num_tokens <= last_k:
        return list(range(num_tokens))
    start = max(0, num_tokens - last_k)
    return list(range(start, num_tokens))


# Residual-Direction Alignment evidence


def compute_attn_evidence_batch(
    h_batch: torch.Tensor,
    W_O: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Residual-Direction Alignment (RDA) evidence for attention.

    For each sample, computes signed per-neuron evidence:
        d_hat = (h @ W_O) / ||h @ W_O||_2      # self-bootstrapped evaluation axis
        s_i   = W_O[:,i] * d_hat               # alignment of neuron i's direction
        e_i   = h_i * s_i                      # signed, activation-weighted evidence

    Args:
        h_batch: [N, d_attn] concatenated attention head outputs (torch.Tensor).
        W_O:     [d_attn, d_model] attention output projection (torch.Tensor).
        eps:     Numerical stability epsilon.

    Returns:
        e_batch: [N, d_attn] signed alignment evidence, values in (-inf, +inf).
    """
    h_tensor = _to_tensor(h_batch)
    W_tensor = _to_tensor(W_O, device=h_tensor.device)

    # Step 1: Δr_attn = h @ W_O → normalize to get evaluation axis d̂
    d_att = h_tensor @ W_tensor  # [N, d_model]
    norms = torch.linalg.norm(d_att, dim=1, keepdim=True)
    d_att_hat = d_att / (norms + eps)  # [N, d_model]

    # Step 2: alignment scores s = W_O^T @ d̂
    s = d_att_hat @ W_tensor.transpose(0, 1)  # [N, d_attn]

    # Step 3: signed evidence e_i = h_i * s_i (no ReLU — preserve sign)
    return h_tensor * s  # [N, d_attn]


def compute_ffn_evidence_batch(
    a_batch: torch.Tensor,
    W_down: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Residual-Direction Alignment (RDA) evidence for FFN.

    For each sample, computes signed per-neuron evidence:
        d_hat = (a @ W_down^T) / ||a @ W_down^T||_2     # evaluation axis
        s_j   = W_down[:,j] * d_hat                     # alignment score
        e_j   = a_j * s_j                              # signed evidence

    Args:
        a_batch: [N, K] FFN intermediate activations (torch.Tensor).
        W_down:  [d_model, K] FFN down projection (torch.Tensor).
        eps:     Numerical stability epsilon.

    Returns:
        e_batch: [N, K] signed alignment evidence, values in (-inf, +inf).
    """
    a_tensor = _to_tensor(a_batch)
    W_tensor = _to_tensor(W_down, device=a_tensor.device)

    # Step 1: Δr_ffn = a @ W_down^T → normalize
    d_ffn = a_tensor @ W_tensor.transpose(0, 1)  # [N, d_model]
    norms = torch.linalg.norm(d_ffn, dim=1, keepdim=True)
    d_ffn_hat = d_ffn / (norms + eps)  # [N, d_model]

    # Step 2: s = W_down @ d̂  (shape: [N, K])
    s = d_ffn_hat @ W_tensor  # [N, K]

    # Step 3: signed evidence e_j = a_j * s_j
    return a_tensor * s  # [N, K]


def init_nig(
    dim: int,
    mu_0: float = 0.0,
    lambda_0: float = 1.0,
    alpha_0: float = 1.0,
    beta_0: float = 1.0,
    device: Optional[torch.device] = None,
) -> NIGPosterior:
    """Factory: create an NIGPosterior with given prior and dimension."""
    return NIGPosterior(
        dim=dim,
        mu_0=mu_0,
        lambda_0=lambda_0,
        alpha_0=alpha_0,
        beta_0=beta_0,
        device=device,
    )
