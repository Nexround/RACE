"""Shared scoring helpers for causal-LM perturbation analyses.

The functions in this module are backend-agnostic: callers provide logits,
token ids, masks, or already-projected logit-lens tensors.  This keeps the
metric semantics testable without loading a model or requiring nnsight/vLLM.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class PPLStats:
    """Accumulated negative-log-likelihood statistics."""

    nll_sum: float
    num_tokens: int

    @property
    def mean_nll(self) -> float:
        return self.nll_sum / self.num_tokens if self.num_tokens else float("inf")

    @property
    def ppl(self) -> float:
        if self.num_tokens == 0:
            return float("inf")
        return float(math.exp(self.mean_nll))


@dataclass(frozen=True)
class KLStats:
    """Accumulated token-level KL statistics."""

    kl_sum: float
    num_tokens: int
    kl_values: Sequence[float]

    @property
    def mean_kl(self) -> float:
        return self.kl_sum / self.num_tokens if self.num_tokens else 0.0

    @property
    def std_kl(self) -> float:
        if len(self.kl_values) <= 1:
            return 0.0
        return float(torch.tensor(self.kl_values, dtype=torch.float64).std().item())

    @property
    def max_kl(self) -> float:
        return float(max(self.kl_values)) if self.kl_values else 0.0


def _as_batched_logits(logits: torch.Tensor) -> torch.Tensor:
    """Normalise logits to ``[batch, seq, vocab]``."""

    if logits.dim() == 1:
        return logits.view(1, 1, -1)
    if logits.dim() == 2:
        return logits.unsqueeze(0)
    if logits.dim() == 3:
        return logits
    raise ValueError(f"Expected logits with 1-3 dims, got shape {tuple(logits.shape)}")


def _shift_for_next_token(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return logits/labels/mask for standard next-token scoring."""

    logits = _as_batched_logits(logits)
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    if attention_mask.dim() == 1:
        attention_mask = attention_mask.unsqueeze(0)
    if logits.shape[:2] != input_ids.shape:
        raise ValueError(
            "Logits and input_ids shape mismatch: "
            f"logits={tuple(logits.shape)}, input_ids={tuple(input_ids.shape)}"
        )
    if input_ids.shape != attention_mask.shape:
        raise ValueError(
            "input_ids and attention_mask shape mismatch: "
            f"input_ids={tuple(input_ids.shape)}, mask={tuple(attention_mask.shape)}"
        )
    if logits.shape[1] < 2:
        empty_logits = logits[:, :0, :]
        empty_labels = input_ids[:, :0]
        empty_mask = attention_mask[:, :0]
        return empty_logits, empty_labels, empty_mask
    return logits[:, :-1, :], input_ids[:, 1:], attention_mask[:, 1:]


def compute_shifted_ppl_stats(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> PPLStats:
    """Compute shifted next-token NLL/PPL stats for one logit tensor."""

    shift_logits, shift_labels, shift_mask = _shift_for_next_token(
        logits, input_ids, attention_mask
    )
    if shift_labels.numel() == 0:
        return PPLStats(nll_sum=0.0, num_tokens=0)

    shift_logits = shift_logits.contiguous()
    shift_labels = shift_labels.contiguous()
    shift_mask = shift_mask.to(dtype=shift_logits.dtype).contiguous()
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.shape[-1]),
        shift_labels.view(-1),
        reduction="none",
    ).view_as(shift_labels)
    masked = loss * shift_mask
    return PPLStats(
        nll_sum=float(masked.sum().item()),
        num_tokens=int(shift_mask.sum().item()),
    )


def compute_next_token_ppl_stats(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
) -> PPLStats:
    """Compute NLL/PPL stats for independent next-token predictions.

    Args:
        logits: ``[N, vocab]`` or ``[vocab]`` logits.
        target_ids: ``[N]`` target token ids.
    """

    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
    if target_ids.dim() == 0:
        target_ids = target_ids.unsqueeze(0)
    if logits.shape[0] != target_ids.shape[0]:
        raise ValueError(
            f"Batch mismatch: logits={tuple(logits.shape)}, targets={tuple(target_ids.shape)}"
        )
    loss = F.cross_entropy(logits, target_ids, reduction="sum")
    return PPLStats(nll_sum=float(loss.item()), num_tokens=int(target_ids.numel()))


def compute_shifted_kl_stats(
    baseline_logits: torch.Tensor,
    perturbed_logits: torch.Tensor,
    attention_mask: torch.Tensor,
    direction: str = "forward",
) -> KLStats:
    """Compute shifted token-level KL stats over a teacher-forced sequence."""

    baseline_logits = _as_batched_logits(baseline_logits)
    perturbed_logits = _as_batched_logits(perturbed_logits)
    if baseline_logits.shape != perturbed_logits.shape:
        raise ValueError(
            "Baseline and perturbed logits must have the same shape: "
            f"{tuple(baseline_logits.shape)} vs {tuple(perturbed_logits.shape)}"
        )
    dummy_ids = torch.zeros(
        baseline_logits.shape[:2], dtype=torch.long, device=baseline_logits.device
    )
    shift_base, _, shift_mask = _shift_for_next_token(
        baseline_logits, dummy_ids, attention_mask
    )
    shift_pert, _, _ = _shift_for_next_token(
        perturbed_logits, dummy_ids, attention_mask
    )
    return compute_positionwise_kl_stats(shift_base, shift_pert, shift_mask, direction)


def compute_positionwise_kl_stats(
    baseline_logits: torch.Tensor,
    perturbed_logits: torch.Tensor,
    mask: torch.Tensor | None = None,
    direction: str = "forward",
) -> KLStats:
    """Compute KL stats for aligned next-token distributions."""

    baseline_logits = _as_batched_logits(baseline_logits)
    perturbed_logits = _as_batched_logits(perturbed_logits)
    if baseline_logits.shape != perturbed_logits.shape:
        raise ValueError(
            "Baseline and perturbed logits must have the same shape: "
            f"{tuple(baseline_logits.shape)} vs {tuple(perturbed_logits.shape)}"
        )
    if direction not in ("forward", "reverse"):
        raise ValueError("direction must be 'forward' or 'reverse'")

    log_p = F.log_softmax(baseline_logits, dim=-1)
    log_q = F.log_softmax(perturbed_logits, dim=-1)
    if direction == "reverse":
        log_p, log_q = log_q, log_p

    p = log_p.exp()
    kl = (p * (log_p - log_q)).sum(dim=-1)
    if mask is None:
        mask_bool = torch.ones_like(kl, dtype=torch.bool)
    else:
        if mask.dim() == 1:
            mask = mask.unsqueeze(0)
        if mask.shape != kl.shape:
            raise ValueError(
                f"Mask shape {tuple(mask.shape)} does not match KL shape {tuple(kl.shape)}"
            )
        mask_bool = mask.bool()
    values = kl[mask_bool].detach().cpu().to(torch.float64)
    return KLStats(
        kl_sum=float(values.sum().item()),
        num_tokens=int(values.numel()),
        kl_values=[float(x) for x in values.tolist()],
    )


def compute_target_ranks(
    logits: torch.Tensor, target_ids: torch.Tensor
) -> torch.Tensor:
    """Return 0-indexed rank of each target id in each logit distribution."""

    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
    if target_ids.dim() == 0:
        target_ids = target_ids.unsqueeze(0)
    if logits.shape[0] != target_ids.shape[0]:
        raise ValueError(
            f"Batch mismatch: logits={tuple(logits.shape)}, targets={tuple(target_ids.shape)}"
        )
    target_scores = logits.gather(1, target_ids.view(-1, 1))
    return (logits > target_scores).sum(dim=-1).to(torch.long)


def summarize_rank_rows(rank_rows: Iterable[torch.Tensor]) -> Dict[str, List[float]]:
    """Summarise rows of layerwise ranks into mean/median/std curves."""

    rows = [row.detach().cpu().to(torch.float64) for row in rank_rows]
    if not rows:
        return {"mean_rank": [], "median_rank": [], "std_rank": []}
    arr = torch.stack(rows)
    return {
        "mean_rank": arr.mean(dim=0).tolist(),
        "median_rank": torch.quantile(arr, 0.5, dim=0).tolist(),
        "std_rank": arr.std(dim=0, unbiased=False).tolist(),
    }


def combine_ppl_stats(stats: Iterable[PPLStats]) -> PPLStats:
    """Combine multiple PPL accumulators."""

    stats = list(stats)
    return PPLStats(
        nll_sum=sum(s.nll_sum for s in stats),
        num_tokens=sum(s.num_tokens for s in stats),
    )


def combine_kl_stats(stats: Iterable[KLStats]) -> KLStats:
    """Combine multiple KL accumulators."""

    stats = list(stats)
    values: List[float] = []
    for s in stats:
        values.extend(float(v) for v in s.kl_values)
    return KLStats(
        kl_sum=sum(s.kl_sum for s in stats),
        num_tokens=sum(s.num_tokens for s in stats),
        kl_values=values,
    )
