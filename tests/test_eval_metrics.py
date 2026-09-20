"""Unit tests for PPL and KL divergence evaluation metrics.

Tests the core computation functions with synthetic logits/labels
to verify numerical correctness without requiring a real model.
"""

import math
from argparse import Namespace

import pytest
import torch
import torch.nn.functional as F

from race_eval.llm.evaluation.ppl_eval import compute_perplexity
from race_eval.llm.evaluation.kl_eval import (
    compute_kl_per_batch,
    compute_kl_stats_per_batch,
)
from race_eval.llm.evaluation.utils import pad_and_batch, validate_common_args

# ======================================================================
# PPL tests
# ======================================================================


class TestComputePerplexity:
    def test_uniform_distribution(self):
        """PPL of uniform distribution over V tokens should be V."""
        vocab_size = 100
        seq_len = 20
        batch = 2

        logits = torch.zeros(batch, seq_len, vocab_size)
        input_ids = torch.randint(0, vocab_size, (batch, seq_len))
        attention_mask = torch.ones(batch, seq_len)

        ppl = compute_perplexity([(logits, input_ids, attention_mask)])
        assert abs(ppl - vocab_size) < 1.0, f"Expected ~{vocab_size}, got {ppl}"

    def test_perfect_prediction(self):
        """PPL should be ~1 when the model assigns all probability to the correct token."""
        vocab_size = 50
        seq_len = 10
        batch = 1

        input_ids = torch.randint(0, vocab_size, (batch, seq_len))
        logits = torch.full((batch, seq_len, vocab_size), -100.0)
        for b in range(batch):
            for t in range(seq_len - 1):
                logits[b, t, input_ids[b, t + 1]] = 100.0

        attention_mask = torch.ones(batch, seq_len)
        ppl = compute_perplexity([(logits, input_ids, attention_mask)])
        assert ppl < 1.01, f"Expected ~1.0, got {ppl}"

    def test_masking_excludes_padding(self):
        """Padding positions should not affect PPL calculation.

        With left-padding mask ``[0,0,0,1,1,...,1]``, the shifted mask
        ``mask[:, 1:]`` becomes ``[0,0,1,1,...,1]``.  Logit at position
        ``t`` predicts the token at position ``t+1``, and is weighted by
        ``mask[t+1]``.  So we must corrupt only positions whose
        *successor* is also masked.
        """
        vocab_size = 50
        seq_len = 10
        batch = 1
        pad_len = 4  # first 4 positions are padding

        input_ids = torch.randint(0, vocab_size, (batch, seq_len))
        logits = torch.full((batch, seq_len, vocab_size), -100.0)
        for b in range(batch):
            for t in range(seq_len - 1):
                logits[b, t, input_ids[b, t + 1]] = 100.0

        mask_partial = torch.ones(batch, seq_len)
        mask_partial[:, :pad_len] = 0
        # Corrupt logits at positions 0..pad_len-2 (safe: their shifted
        # label positions 1..pad_len-1 are all within the mask=0 region).
        logits[:, : pad_len - 1, :] = 0.0
        ppl_masked = compute_perplexity([(logits, input_ids, mask_partial)])

        assert ppl_masked < 1.01, f"Masked PPL should be ~1.0, got {ppl_masked}"

    def test_multiple_batches(self):
        """PPL computation should handle multiple batches correctly."""
        vocab_size = 50
        batch_results = []
        for _ in range(3):
            logits = torch.zeros(2, 8, vocab_size)
            input_ids = torch.randint(0, vocab_size, (2, 8))
            mask = torch.ones(2, 8)
            batch_results.append((logits, input_ids, mask))

        ppl = compute_perplexity(batch_results)
        assert abs(ppl - vocab_size) < 1.0

    def test_empty_returns_inf(self):
        """Empty input should return inf."""
        ppl = compute_perplexity([])
        assert ppl == float("inf")


# ======================================================================
# KL divergence tests
# ======================================================================


class TestComputeKLDivergence:
    def test_identical_distributions_zero_kl(self):
        """KL divergence between identical distributions should be 0."""
        logits = torch.randn(2, 10, 50)
        mask = torch.ones(2, 10)

        sum_kl, n_tok = compute_kl_per_batch(logits, logits, mask)
        mean_kl = sum_kl / max(n_tok, 1)
        assert abs(mean_kl) < 1e-5, f"Expected ~0, got {mean_kl}"

    def test_different_distributions_positive_kl(self):
        """KL divergence between different distributions should be positive."""
        logits_a = torch.randn(2, 10, 50)
        logits_b = torch.randn(2, 10, 50) + 5.0  # shifted distribution
        mask = torch.ones(2, 10)

        sum_kl, n_tok = compute_kl_per_batch(logits_a, logits_b, mask)
        mean_kl = sum_kl / max(n_tok, 1)
        assert mean_kl > 0, f"Expected positive KL, got {mean_kl}"

    def test_masking_excludes_padding(self):
        """Masked (padding) positions should not contribute to KL."""
        logits_a = torch.randn(1, 10, 50)
        logits_b = torch.randn(1, 10, 50) + 10.0

        mask_full = torch.ones(1, 10)
        sum_full, n_full = compute_kl_per_batch(logits_a, logits_b, mask_full)

        mask_half = torch.ones(1, 10)
        mask_half[:, :5] = 0  # mask out first 5 positions
        sum_half, n_half = compute_kl_per_batch(logits_a, logits_b, mask_half)

        assert n_half < n_full
        # Mean KL should differ because different positions have different values
        assert sum_half < sum_full

    def test_reverse_direction(self):
        """KL(P||Q) != KL(Q||P) in general (asymmetry)."""
        logits_a = torch.randn(2, 10, 50)
        logits_b = torch.randn(2, 10, 50) + 3.0
        mask = torch.ones(2, 10)

        sum_fwd, _ = compute_kl_per_batch(logits_a, logits_b, mask, "forward")
        sum_rev, _ = compute_kl_per_batch(logits_a, logits_b, mask, "reverse")

        # They should both be positive but generally different
        assert sum_fwd > 0
        assert sum_rev > 0
        assert abs(sum_fwd - sum_rev) > 1e-4, "KL should be asymmetric"

    def test_stats_returns_per_token_values(self):
        """compute_kl_stats_per_batch should return one value per valid token."""
        logits_a = torch.randn(2, 8, 30)
        logits_b = torch.randn(2, 8, 30)
        mask = torch.ones(2, 8)
        mask[1, 6:] = 0  # mask last 2 of second sample

        values = compute_kl_stats_per_batch(logits_a, logits_b, mask)
        expected_count = int(mask.sum().item())
        assert len(values) == expected_count


# ======================================================================
# pad_and_batch tests
# ======================================================================


class TestPadAndBatch:
    def test_basic_batching(self):
        """Should group tensors into correctly sized batches."""
        tensors = [torch.arange(5), torch.arange(8), torch.arange(3)]
        batches = pad_and_batch(tensors, pad_token_id=0, batch_size=2)

        assert len(batches) == 2
        ids_0, mask_0 = batches[0]
        assert ids_0.shape == (2, 8)  # padded to max in batch
        assert mask_0.shape == (2, 8)

        ids_1, mask_1 = batches[1]
        assert ids_1.shape == (1, 3)
        assert mask_1.shape == (1, 3)

    def test_left_padding(self):
        """Shorter sequences should be left-padded."""
        short = torch.tensor([10, 20, 30])
        long = torch.tensor([1, 2, 3, 4, 5])
        batches = pad_and_batch([short, long], pad_token_id=0, batch_size=2)

        ids, mask = batches[0]
        # Short sequence: [0, 0, 10, 20, 30] (left-padded)
        assert ids[0, 0].item() == 0
        assert ids[0, 1].item() == 0
        assert ids[0, 2].item() == 10
        # Mask: [0, 0, 1, 1, 1]
        assert mask[0, 0].item() == 0
        assert mask[0, 2].item() == 1

    def test_single_element(self):
        """Single tensor should produce one batch with no padding."""
        t = torch.tensor([1, 2, 3])
        batches = pad_and_batch([t], pad_token_id=0, batch_size=4)
        assert len(batches) == 1
        ids, mask = batches[0]
        assert ids.shape == (1, 3)
        assert mask.sum().item() == 3


# ======================================================================
# CLI argument validation tests
# ======================================================================


class TestEvalArgumentValidation:
    def test_direct_model_comparison_requires_both_paths(self):
        args = Namespace(
            model=None,
            race_h5=None,
            baseline_model="base",
            perturbed_model=None,
        )

        with pytest.raises(ValueError, match="provided together"):
            validate_common_args(args)

    def test_direct_model_comparison_does_not_require_h5(self):
        args = Namespace(
            model=None,
            race_h5=None,
            baseline_model="base",
            perturbed_model="perturbed",
        )

        assert validate_common_args(args) is True

    def test_hook_mode_still_requires_race_h5(self):
        args = Namespace(
            model="base",
            race_h5=None,
            baseline_model=None,
            perturbed_model=None,
        )

        with pytest.raises(ValueError, match="--race-h5 is required"):
            validate_common_args(args)
