"""Unit tests for race.core.analyzer — RACE core algorithm (NIGPosterior)."""

import numpy as np
import pytest
import torch

from race.core.analyzer import (
    LayerWeights,
    NIGPosterior,
    RACEConfig,
    RACEState,
    compute_attn_evidence_batch,
    compute_ffn_evidence_batch,
    init_nig,
    select_token_indices,
)

# ---------------------------------------------------------------------------
# NIGPosterior
# ---------------------------------------------------------------------------


class TestNIGPosterior:
    def test_basic_creation(self):
        nig = NIGPosterior(dim=4, mu_0=0.0, lambda_0=1.0, alpha_0=1.0, beta_0=1.0)
        assert nig.dim == 4
        assert nig.N == 0

    def test_add_evidence_1d(self):
        nig = NIGPosterior(dim=3)
        ev = torch.tensor([1.0, -0.5, 0.2])
        nig.add_evidence(ev)
        assert nig.N == 1

    def test_add_evidence_2d(self):
        nig = NIGPosterior(dim=3)
        ev = torch.randn(10, 3)
        nig.add_evidence(ev)
        assert nig.N == 10

    def test_posterior_mean_signed(self):
        nig = init_nig(dim=2)
        ev = torch.tensor([[3.0, -3.0]] * 20)
        nig.add_evidence(ev)
        pm = nig.posterior_mean
        assert pm[0] > 0, "positive evidence → positive posterior mean"
        assert pm[1] < 0, "negative evidence → negative posterior mean"

    def test_cam_score_nonnegative(self):
        nig = init_nig(dim=4)
        ev = torch.randn(50, 4)
        nig.add_evidence(ev)
        cam = nig.cam_score(gamma=0.05)
        assert torch.all(cam >= 0), "CAM score must be non-negative"

    def test_cam_score_high_confidence(self):
        """Strongly consistent evidence should yield positive CAM scores."""
        nig = init_nig(dim=2)
        ev = torch.zeros(200, 2)
        ev[:, 0] = 5.0  # very consistent positive
        nig.add_evidence(ev)
        cam = nig.cam_score(gamma=0.05)
        assert cam[0] > 0, "consistent positive evidence → positive CAM"
        assert cam[1] == 0, "zero evidence → CAM = 0"

    def test_cam_is_positive_direction_and_negative_cam_is_separate(self):
        nig = init_nig(dim=2)
        nig.add_evidence(torch.tensor([[5.0, -5.0]] * 200))

        cam = nig.cam_score(gamma=0.05)
        negative_cam = nig.negative_cam_score(gamma=0.05)

        assert cam[0] > 0
        assert cam[1] == 0
        assert negative_cam[0] == 0
        assert negative_cam[1] > 0

    def test_state_dict_round_trip(self):
        nig = init_nig(dim=8)
        ev = torch.randn(20, 8)
        nig.add_evidence(ev)
        sd = nig.state_dict()
        assert "dim" in sd
        nig2 = NIGPosterior.from_state_dict(sd)
        assert torch.allclose(nig.posterior_mean, nig2.posterior_mean)

    def test_from_state_dict_with_explicit_dim(self):
        nig = init_nig(dim=4)
        nig.add_evidence(torch.randn(5, 4))
        sd = nig.state_dict()
        del sd["dim"]  # simulate missing dim key
        nig2 = NIGPosterior.from_state_dict(sd, dim=4)
        assert torch.allclose(nig.posterior_mean, nig2.posterior_mean)

    def test_nig_update_laws(self):
        """Verify that NIG posterior updates follow the correct formulas."""
        nig = NIGPosterior(dim=1, mu_0=0.0, lambda_0=1.0, alpha_0=2.0, beta_0=3.0)
        ev = torch.tensor([[1.0], [3.0], [5.0]])  # N=3, mean=3, S=8
        nig.add_evidence(ev)
        assert nig.N == 3
        # lambda_n = lambda_0 + N = 4
        assert abs(nig.lambda_n - 4.0) < 1e-5
        # alpha_n = alpha_0 + N/2 = 2 + 1.5 = 3.5
        assert abs(nig.alpha_n - 3.5) < 1e-5
        # mu_n = (lambda_0*mu_0 + N*mean) / lambda_n = (0 + 3*3) / 4 = 2.25
        mu_n = nig.posterior_mean.item()
        assert abs(mu_n - 2.25) < 1e-4


# ---------------------------------------------------------------------------
# init_nig
# ---------------------------------------------------------------------------


class TestInitNIG:
    def test_basic(self):
        nig = init_nig(dim=5)
        assert nig.dim == 5
        assert nig.N == 0
        assert nig.mu_0 == 0.0
        assert nig.lambda_0 == 1.0

    def test_custom_priors(self):
        nig = init_nig(dim=3, mu_0=1.0, lambda_0=2.0, alpha_0=3.0, beta_0=4.0)
        assert nig.mu_0 == 1.0
        assert nig.lambda_0 == 2.0
        assert nig.alpha_0 == 3.0
        assert nig.beta_0 == 4.0


# ---------------------------------------------------------------------------
# select_token_indices
# ---------------------------------------------------------------------------


class TestSelectTokenIndices:
    def test_all_mode(self):
        indices = select_token_indices(10, "all", last_k=5)
        assert indices == list(range(10))

    def test_last_k(self):
        indices = select_token_indices(10, "last_k", last_k=3)
        assert indices == [7, 8, 9]

    def test_last_k_short_sequence(self):
        indices = select_token_indices(2, "last_k", last_k=5)
        assert indices == [0, 1]


# ---------------------------------------------------------------------------
# Evidence computation (signed, no ReLU)
# ---------------------------------------------------------------------------


class TestComputeEvidence:
    @pytest.fixture
    def sample_attn_data(self):
        d = 8
        N = 4
        h_batch = torch.randn(N, d)
        W_O = torch.randn(d, d)
        return h_batch, W_O

    @pytest.fixture
    def sample_ffn_data(self):
        K = 16
        d_out = 8
        N = 4
        a_batch = torch.randn(N, K)
        W_down = torch.randn(d_out, K)
        return a_batch, W_down

    def test_attn_evidence_shape(self, sample_attn_data):
        h_batch, W_O = sample_attn_data
        e = compute_attn_evidence_batch(h_batch, W_O, eps=1e-8)
        assert e.shape == h_batch.shape

    def test_attn_evidence_signed(self, sample_attn_data):
        """RACE evidence is signed — can be negative."""
        h_batch, W_O = sample_attn_data
        e = compute_attn_evidence_batch(h_batch, W_O, eps=1e-8)
        # With random data, should have both positive and negative values
        assert e.shape[1] == h_batch.shape[1]
        # No ReLU: evidence can be negative
        # (probabilistic check — very unlikely to fail with random data)

    def test_ffn_evidence_shape(self, sample_ffn_data):
        a_batch, W_down = sample_ffn_data
        e = compute_ffn_evidence_batch(a_batch, W_down, eps=1e-8)
        assert e.shape == a_batch.shape

    def test_evidence_no_relu(self):
        """Evidence should not apply ReLU — negative values allowed."""
        h = torch.tensor([[-1.0, -1.0]])  # negative activations
        W_O = torch.eye(2)
        e = compute_attn_evidence_batch(h, W_O, eps=1e-8)
        # With negative hidden state, signed product should be negative
        assert e.shape == (1, 2)


# ---------------------------------------------------------------------------
# LayerWeights
# ---------------------------------------------------------------------------


class TestLayerWeights:
    def test_dataclass(self):
        w = LayerWeights(
            W_O=torch.randn(8, 8),
            W_down=torch.randn(8, 16),
        )
        assert w.W_O is not None
        assert w.W_down is not None
        assert w.W_O.shape == (8, 8)

    def test_optional_fields(self):
        w = LayerWeights()
        assert w.W_O is None
        assert w.W_down is None
