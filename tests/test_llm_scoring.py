import math

import pytest
import torch
import torch.nn.functional as F

from race_eval.llm.evaluation.scoring import (
    combine_kl_stats,
    combine_ppl_stats,
    compute_next_token_ppl_stats,
    compute_positionwise_kl_stats,
    compute_shifted_kl_stats,
    compute_shifted_ppl_stats,
    compute_target_ranks,
    summarize_rank_rows,
)


def test_shifted_ppl_uniform_distribution():
    vocab = 17
    logits = torch.zeros(2, 5, vocab)
    input_ids = torch.randint(0, vocab, (2, 5))
    mask = torch.ones(2, 5)

    stats = compute_shifted_ppl_stats(logits, input_ids, mask)

    assert stats.num_tokens == 8
    assert stats.ppl == pytest.approx(vocab, rel=1e-5)


def test_shifted_ppl_respects_successor_mask():
    vocab = 5
    input_ids = torch.tensor([[0, 0, 2, 3]])
    mask = torch.tensor([[0, 0, 1, 1]])
    logits = torch.zeros(1, 4, vocab)
    logits[:, :, :] = -100.0
    logits[0, 0, 1] = 100.0  # ignored: successor mask is 0
    logits[0, 1, 2] = 100.0
    logits[0, 2, 3] = 100.0

    stats = compute_shifted_ppl_stats(logits, input_ids, mask)

    assert stats.num_tokens == 2
    assert stats.ppl == pytest.approx(1.0)


def test_next_token_ppl_matches_cross_entropy():
    logits = torch.tensor([[0.0, 2.0], [3.0, 0.0]])
    targets = torch.tensor([1, 0])

    stats = compute_next_token_ppl_stats(logits, targets)
    expected_nll = F.cross_entropy(logits, targets, reduction="sum").item()

    assert stats.num_tokens == 2
    assert stats.nll_sum == pytest.approx(expected_nll)
    assert stats.ppl == pytest.approx(math.exp(expected_nll / 2))


def test_positionwise_kl_identical_is_zero():
    logits = torch.randn(2, 3, 7)
    stats = compute_positionwise_kl_stats(logits, logits)

    assert stats.num_tokens == 6
    assert stats.mean_kl == pytest.approx(0.0, abs=1e-6)
    assert stats.max_kl == pytest.approx(0.0, abs=1e-6)


def test_shifted_kl_uses_shifted_mask():
    base = torch.zeros(1, 4, 3)
    pert = torch.zeros(1, 4, 3)
    pert[0, 0, 0] = 10.0  # ignored: successor mask is 0
    pert[0, 1, 1] = 5.0
    pert[0, 2, 2] = 5.0
    mask = torch.tensor([[0, 0, 1, 1]])

    stats = compute_shifted_kl_stats(base, pert, mask)

    assert stats.num_tokens == 2
    assert stats.mean_kl > 0


def test_kl_direction_validation():
    with pytest.raises(ValueError, match="direction"):
        compute_positionwise_kl_stats(
            torch.zeros(2, 3), torch.zeros(2, 3), direction="bad"
        )


def test_target_ranks_are_zero_indexed():
    logits = torch.tensor(
        [
            [0.1, 0.9, 0.2],
            [5.0, 1.0, 3.0],
        ]
    )
    targets = torch.tensor([1, 2])

    ranks = compute_target_ranks(logits, targets)

    assert ranks.tolist() == [0, 1]


def test_summarize_rank_rows():
    summary = summarize_rank_rows(
        [
            torch.tensor([0, 10, 20]),
            torch.tensor([2, 14, 26]),
        ]
    )

    assert summary["mean_rank"] == [1.0, 12.0, 23.0]
    assert summary["median_rank"] == [1.0, 12.0, 23.0]
    assert summary["std_rank"] == pytest.approx([1.0, 2.0, 3.0])


def test_combine_stats():
    ppl = combine_ppl_stats(
        [
            compute_next_token_ppl_stats(torch.tensor([[2.0, 0.0]]), torch.tensor([0])),
            compute_next_token_ppl_stats(torch.tensor([[0.0, 2.0]]), torch.tensor([1])),
        ]
    )
    kl = combine_kl_stats(
        [
            compute_positionwise_kl_stats(
                torch.zeros(1, 2), torch.tensor([[1.0, 0.0]])
            ),
            compute_positionwise_kl_stats(
                torch.zeros(1, 2), torch.tensor([[0.0, 1.0]])
            ),
        ]
    )

    assert ppl.num_tokens == 2
    assert ppl.ppl < 2.0
    assert kl.num_tokens == 2
    assert len(kl.kl_values) == 2
