import math

import pytest
import torch

from race.core.analyzer import NIGPosterior
from race.core.online_accumulator import OnlineRACEAccumulator


def test_empirical_snr_uses_sample_standard_deviation():
    posterior = NIGPosterior(dim=2)
    posterior.add_evidence(torch.tensor([[1.0, -1.0], [3.0, -3.0]]))

    expected = 2.0 / math.sqrt(2.0)
    assert posterior.empirical_snr[0].item() == pytest.approx(expected)
    assert posterior.empirical_snr[1].item() == 0.0


def test_activation_mean_accumulates_over_tokens():
    act_sum = {}
    act_n = {}

    OnlineRACEAccumulator._update_act_stats(
        act_sum,
        act_n,
        0,
        torch.tensor([[1.0, 3.0], [3.0, 5.0]]),
        torch.device("cpu"),
    )
    OnlineRACEAccumulator._update_act_stats(
        act_sum,
        act_n,
        0,
        torch.tensor([[5.0, 7.0]]),
        torch.device("cpu"),
    )

    assert act_n[0] == 3
    assert torch.allclose(act_sum[0] / act_n[0], torch.tensor([3.0, 5.0]))


def test_activation_mean_ignores_empty_token_batches():
    act_sum = {}
    act_n = {}

    OnlineRACEAccumulator._update_act_stats(
        act_sum,
        act_n,
        0,
        torch.empty(0, 4),
        torch.device("cpu"),
    )

    assert act_sum == {}
    assert act_n == {}
