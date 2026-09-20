"""Unit tests for race_eval.ablation.plan — ablation plan construction."""

import h5py
import numpy as np
import pytest

from race.io.h5_schema import (
    ensure_axis_value,
    ensure_race_meta,
    ensure_reports_group,
)
from race_eval.ablation.plan import (
    AblationPlan,
    OperationMode,
    build_ablation_plan,
    create_ablation_plan,
    create_reference_filtered_ablation_plan,
    list_llm_h5_layer_indices,
    load_top_k_neurons_from_h5,
    select_bottom_indices,
    select_top_indices,
)

# ---------------------------------------------------------------------------
# OperationMode
# ---------------------------------------------------------------------------


class TestOperationMode:
    def test_values(self):
        assert OperationMode.SUPPRESS.value == "suppress"
        assert OperationMode.ENHANCE.value == "enhance"
        assert OperationMode.KEEP_TOP.value == "keep_top"
        assert OperationMode.BASELINE.value == "baseline"


# ---------------------------------------------------------------------------
# select_top_indices / select_bottom_indices
# ---------------------------------------------------------------------------


class TestSelectIndices:
    def test_top_basic(self):
        scores = np.array([0.1, 0.5, 0.3, 0.05, 0.05])
        indices = select_top_indices(scores, top_percent=40.0)
        # 40% of 5 = 2 neurons
        assert len(indices) == 2
        # Should include the two with highest contributions
        assert 1 in indices  # 0.5
        assert 2 in indices  # 0.3

    def test_top_empty(self):
        assert select_top_indices(np.array([]), top_percent=10.0) == []
        assert select_top_indices(np.array([1.0, 2.0]), top_percent=0.0) == []

    def test_top_minimum_one(self):
        scores = np.array([0.1, 0.9])
        indices = select_top_indices(scores, top_percent=1.0)
        # Even very small percent should select at least 1
        assert len(indices) >= 1

    def test_bottom_basic(self):
        scores = np.array([0.1, 0.5, 0.3, 0.05, 0.05])
        indices = select_bottom_indices(scores, top_percent=40.0)
        # Keep top 40% (2 neurons), return bottom 60% (3 neurons)
        assert len(indices) == 3

    def test_bottom_full(self):
        scores = np.array([1.0, 2.0, 3.0])
        indices = select_bottom_indices(scores, top_percent=0.0)
        assert len(indices) == 3

    def test_bottom_none(self):
        scores = np.array([1.0, 2.0, 3.0])
        indices = select_bottom_indices(scores, top_percent=100.0)
        assert len(indices) == 0


# ---------------------------------------------------------------------------
# AblationPlan
# ---------------------------------------------------------------------------


class TestAblationPlan:
    def test_default_creation(self):
        plan = AblationPlan()
        assert plan.operation_mode == "suppress"
        assert plan.top_percent == 5.0
        assert plan.top_k_count is None
        assert plan.total_neurons == 0

    def test_llm_style(self):
        plan = AblationPlan(
            attn_neurons={0: {1, 2, 3}, 1: {4, 5}},
            mlp_neurons={0: {10, 11}},
        )
        assert plan.total_neurons == 7

    def test_cv_style_for_label(self):
        plan = AblationPlan(
            indices={
                0: {"attn": {0: [1, 2]}, "ffn": {0: [3]}},
                1: {"attn": {0: [4]}},
            }
        )
        label_0 = plan.for_label(0)
        assert 0 in label_0["attn"]
        assert label_0["attn"][0] == [1, 2]
        assert plan.for_label(99) == {}

    def test_operation_mode_enum(self):
        plan = AblationPlan(operation_mode="suppress")
        assert plan.operation_mode_enum == OperationMode.SUPPRESS

        plan2 = AblationPlan(operation_mode="enhance")
        assert plan2.operation_mode_enum == OperationMode.ENHANCE


# ---------------------------------------------------------------------------
# build_ablation_plan (CV pipeline)
# ---------------------------------------------------------------------------


class TestBuildAblationPlan:
    @pytest.fixture
    def sample_data(self):
        """Sample RACE posterior data for 2 labels, 2 layers."""
        alpha_attn = {
            0: {0: np.array([0.1, 0.5, 0.3, 0.1]), 1: np.array([0.2, 0.2, 0.4, 0.2])},
            1: {0: np.array([0.4, 0.1, 0.1, 0.4]), 1: np.array([0.1, 0.6, 0.2, 0.1])},
        }
        alpha_ffn = {
            0: {0: np.array([0.05, 0.8, 0.1, 0.05]), 1: np.array([0.3, 0.3, 0.2, 0.2])},
            1: {0: np.array([0.7, 0.1, 0.1, 0.1]), 1: np.array([0.2, 0.2, 0.3, 0.3])},
        }
        return alpha_attn, alpha_ffn

    def test_basic_build(self, sample_data):
        alpha_attn, alpha_ffn = sample_data
        plan = build_ablation_plan(
            alpha_attn=alpha_attn,
            alpha_ffn=alpha_ffn,
            modules=["attn", "ffn"],
            top_percent=25.0,
        )
        assert isinstance(plan, AblationPlan)
        assert 0 in plan.indices
        assert 1 in plan.indices

    def test_layer_filter(self, sample_data):
        alpha_attn, alpha_ffn = sample_data
        plan = build_ablation_plan(
            alpha_attn=alpha_attn,
            alpha_ffn=alpha_ffn,
            modules=["attn"],
            top_percent=25.0,
            layer_filter=[0],
        )
        for label_idx, label_plan in plan.indices.items():
            if "attn" in label_plan:
                assert all(layer_idx == 0 for layer_idx in label_plan["attn"])

    def test_keep_top_mode(self, sample_data):
        alpha_attn, alpha_ffn = sample_data
        plan = build_ablation_plan(
            alpha_attn=alpha_attn,
            alpha_ffn=alpha_ffn,
            modules=["attn"],
            top_percent=25.0,
            operation_mode="keep_top",
        )
        assert plan.operation_mode == "keep_top"

    def test_single_module(self, sample_data):
        alpha_attn, alpha_ffn = sample_data
        plan = build_ablation_plan(
            alpha_attn=alpha_attn,
            alpha_ffn=alpha_ffn,
            modules=["ffn"],
            top_percent=25.0,
        )
        for label_idx, label_plan in plan.indices.items():
            assert "attn" not in label_plan

    def test_strict_missing_module(self):
        with pytest.raises(ValueError, match="no attention data"):
            build_ablation_plan(
                alpha_attn={},
                alpha_ffn={0: {0: np.array([1.0])}},
                modules=["attn"],
                top_percent=10.0,
                strict=True,
            )

    def test_non_strict_missing_module(self):
        plan = build_ablation_plan(
            alpha_attn={},
            alpha_ffn={0: {0: np.array([1.0, 2.0])}},
            modules=["attn", "ffn"],
            top_percent=50.0,
            strict=False,
        )
        assert isinstance(plan, AblationPlan)


# ---------------------------------------------------------------------------
# LLM H5 plan builders
# ---------------------------------------------------------------------------


def _write_llm_ablation_h5(path):
    with h5py.File(path, "w") as h5f:
        _, axes = ensure_race_meta(
            h5f,
            target_model_family="llm",
            model_name="toy-model",
            device="cpu",
            race_config={},
        )
        ensure_axis_value(
            axes,
            axis_type="concept",
            value_key=7,
            display_name="code_generation",
            raw_value={"label_index": 7},
        )

        reports = ensure_reports_group(h5f)
        report = reports.create_group("axes/value_000007")
        modules = report.create_group("modules")

        attn = modules.create_group("attn_pre_output/layer_00")
        attn.create_dataset("posterior_mean", data=np.asarray([0.1, -0.2, 0.7]))
        attn.create_dataset("cam_score", data=np.asarray([0.4, 0.9, 0.2]))
        attn.create_dataset("negative_cam_score", data=np.asarray([0.0, 0.8, 0.0]))

        mlp = modules.create_group("mlp_pre_down/layer_00")
        mlp.create_dataset("posterior_mean", data=np.asarray([-0.5, 0.3, 0.8]))
        mlp.create_dataset("cam_score", data=np.asarray([0.6, 0.1, 0.7]))
        mlp.create_dataset("negative_cam_score", data=np.asarray([0.5, 0.0, 0.0]))


def _write_rsf_h5(path, concept, layers):
    """Write layer/module metric vectors for paper-RSF tests."""
    with h5py.File(path, "w") as h5f:
        _, axes = ensure_race_meta(
            h5f,
            target_model_family="llm",
            model_name="toy-model",
            device="cpu",
            race_config={},
        )
        ensure_axis_value(
            axes,
            axis_type="concept",
            value_key=1,
            display_name=concept,
            raw_value={"label_index": 1},
        )
        modules = (
            ensure_reports_group(h5f)
            .create_group("axes/value_000001")
            .create_group("modules")
        )
        for module_name, module_layers in layers.items():
            module = modules.require_group(module_name)
            for layer_idx, metrics in module_layers.items():
                layer = module.create_group(f"layer_{layer_idx:02d}")
                for metric_name, values in metrics.items():
                    layer.create_dataset(metric_name, data=np.asarray(values))


class TestLlmH5PlanBuilders:
    def test_lists_layers_through_core_reader(self, tmp_path):
        h5_path = tmp_path / "llm_race.h5"
        _write_llm_ablation_h5(h5_path)

        assert list_llm_h5_layer_indices(str(h5_path), "code_generation") == {0}
        assert list_llm_h5_layer_indices(
            str(h5_path),
            "code_generation",
            modules=["attn_pre_output"],
        ) == {0}
        assert (
            list_llm_h5_layer_indices(
                str(h5_path),
                "code_generation",
                modules=[],
            )
            == set()
        )

    def test_loads_top_k_neurons_through_core_reader(self, tmp_path):
        h5_path = tmp_path / "llm_race.h5"
        _write_llm_ablation_h5(h5_path)

        attn, mlp = load_top_k_neurons_from_h5(
            str(h5_path),
            "code_generation",
            top_k_percent=100.0,
            metric="lcb_pos",
            top_k_count=1,
        )

        assert attn == {0: {0}}
        assert mlp == {0: {2}}

    def test_create_ablation_plan_from_h5(self, tmp_path):
        h5_path = tmp_path / "llm_race.h5"
        _write_llm_ablation_h5(h5_path)

        plan = create_ablation_plan(
            h5_path=str(h5_path),
            concept_name="code_generation",
            operation_mode=OperationMode.SUPPRESS,
            top_k_percent=50.0,
            metric="posterior_mean",
            modules=["mlp"],
            top_k_count=1,
        )

        assert plan.operation_mode == "suppress"
        assert plan.attn_neurons == {}
        assert plan.mlp_neurons == {0: {2}}

    def test_negative_cam_reads_separate_dataset(self, tmp_path):
        h5_path = tmp_path / "llm_race.h5"
        _write_llm_ablation_h5(h5_path)

        attn, mlp = load_top_k_neurons_from_h5(
            str(h5_path),
            "code_generation",
            top_k_percent=100.0,
            metric="lcb_neg",
            top_k_count=1,
        )

        assert attn == {0: {1}}
        assert mlp == {0: {0}}

    def test_zero_percent_selects_no_neurons(self, tmp_path):
        h5_path = tmp_path / "llm_race.h5"
        _write_llm_ablation_h5(h5_path)

        attn, mlp = load_top_k_neurons_from_h5(
            str(h5_path),
            "code_generation",
            top_k_percent=0.0,
            metric="lcb_pos",
        )

        assert attn == {}
        assert mlp == {}


class TestReferenceSetFiltering:
    def test_traverses_full_target_ranking_and_preserves_k(self, tmp_path):
        target = tmp_path / "target.h5"
        reference = tmp_path / "reference.h5"
        _write_rsf_h5(
            target,
            "target",
            {
                "attn_pre_output": {
                    0: {
                        "posterior_mean": [5, 4, 3, 2, 1],
                        "cam_score": [10, 9, 8, 7, 6],
                    }
                },
                "mlp_pre_down": {
                    0: {
                        "posterior_mean": [1, 2, 3, 4, 5],
                        "cam_score": [1, 2, 3, 4, 5],
                    }
                },
            },
        )
        _write_rsf_h5(
            reference,
            "reference",
            {
                "attn_pre_output": {
                    0: {
                        "posterior_mean": [1, 5, 4, 3, 2],
                        "cam_score": [1, 10, 9, 2, 0],
                    }
                },
                "mlp_pre_down": {
                    0: {
                        "posterior_mean": [5, 1, 2, 3, 4],
                        "cam_score": [10, 1, 2, 3, 9],
                    }
                },
            },
        )

        plan = create_reference_filtered_ablation_plan(
            str(target),
            "target",
            str(reference),
            "reference",
            OperationMode.SUPPRESS,
            top_k_percent=100.0,
            top_k_count=2,
            metric="lcb_pos",
        )

        # Reference Top-2 excludes attn {1, 2} and MLP {0, 4}. RSF keeps
        # scanning the complete target rankings until exactly two remain.
        assert plan.attn_neurons == {0: {0, 3}}
        assert plan.mlp_neurons == {0: {2, 3}}

    def test_keeps_layer_local_exclusion_sets_isolated(self, tmp_path):
        target = tmp_path / "target.h5"
        reference = tmp_path / "reference.h5"
        target_layers = {
            0: {"posterior_mean": [3, 2, 1], "cam_score": [3, 2, 1]},
            1: {"posterior_mean": [3, 2, 1], "cam_score": [3, 2, 1]},
        }
        reference_layers = {
            0: {"posterior_mean": [3, 2, 1], "cam_score": [3, 2, 1]},
            1: {"posterior_mean": [1, 2, 3], "cam_score": [1, 2, 3]},
        }
        _write_rsf_h5(
            target,
            "target",
            {"attn_pre_output": target_layers},
        )
        _write_rsf_h5(
            reference,
            "reference",
            {"attn_pre_output": reference_layers},
        )

        plan = create_reference_filtered_ablation_plan(
            str(target),
            "target",
            str(reference),
            "reference",
            OperationMode.SUPPRESS,
            top_k_percent=100.0,
            top_k_count=1,
            metric="lcb_pos",
            modules=["attn"],
        )

        assert plan.attn_neurons == {0: {1}, 1: {0}}

    def test_uses_distinct_reference_budget_and_same_metric(self, tmp_path):
        target = tmp_path / "target.h5"
        reference = tmp_path / "reference.h5"
        target_metrics = {
            "posterior_mean": [4, 3, 2, 1],
            "cam_score": [4, 3, 2, 1],
            "empirical_mean": [4, 3, 2, 1],
        }
        reference_metrics = {
            "posterior_mean": [1, 4, 3, 2],
            "cam_score": [1, 4, 3, 2],
            "empirical_mean": [10, 1, 2, 3],
        }
        _write_rsf_h5(
            target,
            "target",
            {"attn_pre_output": {0: target_metrics}},
        )
        _write_rsf_h5(
            reference,
            "reference",
            {"attn_pre_output": {0: reference_metrics}},
        )

        plan = create_reference_filtered_ablation_plan(
            str(target),
            "target",
            str(reference),
            "reference",
            OperationMode.SUPPRESS,
            top_k_percent=100.0,
            top_k_count=2,
            reference_top_k_count=1,
            metric="empirical_mean",
            modules=["attn"],
        )

        # Empirical-mean reference Top-1 is neuron 0. A hard-coded CAM metric
        # would instead exclude neuron 1.
        assert plan.attn_neurons == {0: {1, 2}}

    def test_returns_available_candidates_when_exclusion_leaves_too_few(
        self, tmp_path, caplog
    ):
        target = tmp_path / "target.h5"
        reference = tmp_path / "reference.h5"
        metrics = {
            "posterior_mean": [3, 2, 1],
            "cam_score": [3, 2, 1],
        }
        _write_rsf_h5(
            target,
            "target",
            {"attn_pre_output": {0: metrics}},
        )
        _write_rsf_h5(
            reference,
            "reference",
            {"attn_pre_output": {0: metrics}},
        )

        plan = create_reference_filtered_ablation_plan(
            str(target),
            "target",
            str(reference),
            "reference",
            OperationMode.SUPPRESS,
            top_k_percent=100.0,
            top_k_count=2,
            reference_top_k_count=2,
            metric="lcb_pos",
            modules=["attn"],
        )

        assert plan.attn_neurons == {0: {2}}
        assert "selected 1/2 neurons" in caplog.text
