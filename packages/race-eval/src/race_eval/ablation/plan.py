"""Unified ablation plan construction for neuron modification experiments.

This module provides dataclasses and functions for constructing plans
that specify which neurons to modify (suppress or enhance) based on
RACE posterior distributions. It serves both CV and LLM pipelines.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class OperationMode(Enum):
    """Neuron modification operation modes."""

    SUPPRESS = "suppress"
    ENHANCE = "enhance"
    KEEP_TOP = "keep_top"
    BASELINE = "baseline"
    MEAN_ABLATION = "mean_ablation"


@dataclass
class AblationPlan:
    """Unified ablation plan for both CV and LLM pipelines.

    For CV (label-conditioned):
        Use ``indices`` to store per-label neuron selections:
        ``indices[label_idx][module_name][layer_idx] = [neuron_indices]``

    For LLM (concept-conditioned):
        Use ``attn_neurons`` / ``mlp_neurons`` flat mappings:
        ``attn_neurons[layer_idx] = {neuron_indices}``

    Attributes:
        operation_mode: How to modify neurons (suppress/enhance/keep_top/baseline)
        top_percent: Percentage of top neurons selected
        enhancement_factor: Multiplication factor for ENHANCE mode
        modules: Module names involved (e.g. ["attn", "ffn"] or ["attn", "mlp"])
        layers: Optional layer filter
        indices: CV-style nested dict — indices[label][module][layer] = [neurons]
        attn_neurons: LLM-style flat dict — layer_idx -> set of neuron indices
        mlp_neurons: LLM-style flat dict — layer_idx -> set of neuron indices
        concept_name: Optional concept/label name for identification
        top_k_count: When set (LLM H5 path), select this many neurons per layer/module
            instead of ``top_percent``.
    """

    operation_mode: str = "suppress"
    top_percent: float = 5.0
    top_k_count: Optional[int] = None
    enhancement_factor: float = 2.0
    modules: Sequence[str] = ("attn", "ffn")
    layers: Optional[Sequence[int]] = None

    # CV-style: per-label indices
    indices: Dict[int, Dict[str, Dict[int, List[int]]]] = field(default_factory=dict)

    # LLM-style: flat layer -> neurons
    attn_neurons: Dict[int, Set[int]] = field(default_factory=dict)
    mlp_neurons: Dict[int, Set[int]] = field(default_factory=dict)

    concept_name: Optional[str] = None

    # --- CV helpers ---

    def for_label(self, label_idx: int) -> Dict[str, Dict[int, List[int]]]:
        """Get the ablation plan for a specific label (CV pipeline)."""
        return self.indices.get(label_idx, {})

    # --- LLM helpers ---

    @property
    def total_neurons(self) -> int:
        """Total number of neurons to be modified (LLM pipeline)."""
        attn_count = sum(len(idx) for idx in self.attn_neurons.values())
        mlp_count = sum(len(idx) for idx in self.mlp_neurons.values())
        return attn_count + mlp_count

    @property
    def operation_mode_enum(self) -> OperationMode:
        """Return operation_mode as an OperationMode enum."""
        if isinstance(self.operation_mode, OperationMode):
            return self.operation_mode
        return OperationMode(self.operation_mode)


# ---------------------------------------------------------------------------
# Neuron selection utilities
# ---------------------------------------------------------------------------


def select_top_indices(alpha_vector: np.ndarray, top_percent: float) -> List[int]:
    """Select indices of top-k neurons based on their contributions.

    Args:
        alpha_vector: Array of neuron importance scores
        top_percent: Percentage of neurons to select (0-100)

    Returns:
        Sorted list of selected neuron indices
    """
    if alpha_vector.size == 0 or top_percent <= 0:
        return []
    total = max(int(round(alpha_vector.size * (top_percent / 100.0))), 1)
    order = np.argsort(np.abs(alpha_vector))
    top = order[-total:]
    return sorted(int(idx) for idx in top)


def select_bottom_indices(alpha_vector: np.ndarray, top_percent: float) -> List[int]:
    """Select bottom neurons (complement of top-k), used for keep_top mode.

    Args:
        alpha_vector: Array of neuron importance scores
        top_percent: Percentage of top neurons to keep (0-100)

    Returns:
        Sorted list of bottom neuron indices
    """
    if alpha_vector.size == 0:
        return []
    if top_percent <= 0:
        return list(range(alpha_vector.size))
    if top_percent >= 100:
        return []

    total_top = max(int(round(alpha_vector.size * (top_percent / 100.0))), 1)
    contributions = alpha_vector / (alpha_vector.sum() + 1e-12)
    order = np.argsort(contributions)
    bottom = order[:-total_top] if total_top < len(order) else []
    return sorted(int(idx) for idx in bottom)


def select_top_positive_indices(
    score_vector: np.ndarray,
    posterior_mean_vector: np.ndarray,
    top_percent: float,
) -> List[int]:
    """Select top-k% neurons from positive-mean candidates, ranked by score.

    Implements the three-step positive-only suppression strategy:
      1. **Filter**: keep only neurons where ``posterior_mean > 0`` (positive
         contributors).
      2. **Rank**: sort those neurons by ``score_vector`` (e.g. cam_score)
         in **descending** order.
      3. **Select**: take the top ``top_percent``% of the filtered set.

    Args:
        score_vector: Per-neuron scores used for ranking (e.g. cam_score).
        posterior_mean_vector: Per-neuron posterior means used as filter mask.
        top_percent: Percentage of positive-mean neurons to select (0-100).

    Returns:
        Sorted list of selected neuron indices.
    """
    if score_vector.size == 0 or top_percent <= 0:
        return []

    # Step 1 – keep only neurons with positive posterior mean
    positive_indices = np.where(posterior_mean_vector > 0)[0]
    if positive_indices.size == 0:
        return []

    # Step 2 – rank retained neurons by score descending
    positive_scores = score_vector[positive_indices]
    k = max(int(round(positive_indices.size * (top_percent / 100.0))), 1)
    # argsort is ascending; take the last k entries for the highest scores
    order = np.argsort(positive_scores)
    top_within_positive = order[-k:]
    selected = positive_indices[top_within_positive]

    return sorted(int(idx) for idx in selected)


def select_top_negative_indices(
    score_vector: np.ndarray,
    posterior_mean_vector: np.ndarray,
    top_percent: float,
) -> List[int]:
    """Select top-k% neurons from *negative*-mean candidates, ranked by score.

    Implements the "guardrail suppression" strategy used in Protocol 2:

      1. **Filter**: keep only neurons where ``posterior_mean < 0`` (inhibitory
         neurons that act as brakes / guardrails for the target class).
      2. **Rank**: sort those neurons by ``score_vector`` (e.g. cam_score)
         in **descending** order — highest-magnitude inhibitors first.
      3. **Select**: take the top ``top_percent``% of the filtered set.

    Suppressing these neurons removes the inhibitory signal that prevents
    over-activation of the target class, which is expected to raise the
    false-positive rate for that class without substantially hurting recall.

    Args:
        score_vector: Per-neuron scores used for ranking (e.g. cam_score).
        posterior_mean_vector: Per-neuron posterior means used as filter mask.
        top_percent: Percentage of negative-mean neurons to select (0-100).

    Returns:
        Sorted list of selected neuron indices.
    """
    if score_vector.size == 0 or top_percent <= 0:
        return []

    # Step 1 – keep only neurons with strictly negative posterior mean
    negative_indices = np.where(posterior_mean_vector < 0)[0]
    if negative_indices.size == 0:
        return []

    # Step 2 – rank retained neurons by score descending
    negative_scores = score_vector[negative_indices]
    k = max(int(round(negative_indices.size * (top_percent / 100.0))), 1)
    order = np.argsort(negative_scores)
    top_within_negative = order[-k:]
    selected = negative_indices[top_within_negative]

    return sorted(int(idx) for idx in selected)


# ---------------------------------------------------------------------------
# CV plan builder
# ---------------------------------------------------------------------------


def build_ablation_plan(
    alpha_attn: Dict[int, Dict[int, np.ndarray]],
    alpha_ffn: Dict[int, Dict[int, np.ndarray]],
    modules: Sequence[str],
    top_percent: float,
    operation_mode: str = "suppress",
    enhancement_factor: float = 2.0,
    layer_filter: Optional[Sequence[int]] = None,
    strict: bool = True,
    mu_attn: Optional[Dict[int, Dict[int, np.ndarray]]] = None,
    mu_ffn: Optional[Dict[int, Dict[int, np.ndarray]]] = None,
    mean_sign_filter: Optional[str] = None,
) -> AblationPlan:
    """Build an ablation plan from RACE neuron importance scores (CV pipeline).

    Args:
        alpha_attn: Attention neuron scores (cam_score or posterior_mean) per label/layer.
        alpha_ffn: FFN neuron scores (cam_score or posterior_mean) per label/layer.
        modules: Module names to include (e.g., ["attn", "ffn"])
        top_percent: Percentage of top neurons to select per module/layer
        operation_mode: "suppress", "enhance", or "keep_top"
        enhancement_factor: Multiplication factor for enhancement mode
        layer_filter: Optional list of layer indices to restrict to
        strict: If True, raise error when requested module data is missing
        mu_attn: Optional posterior-mean arrays for attention layers used as a
            sign filter. When provided together with ``mu_ffn`` and
            ``mean_sign_filter``, neuron selection switches to a sign-filtered
            strategy: candidates are filtered by posterior-mean sign then ranked
            by the corresponding ``alpha`` score in descending order.
        mu_ffn: Optional posterior-mean arrays for FFN layers (same role as
            ``mu_attn``).
        mean_sign_filter: Optional sign-filter mode for posterior means.
            Accepted values: ``"positive"`` (keep ``mu > 0`` candidates) or
            ``"negative"`` (keep ``mu < 0`` candidates).

    Returns:
        AblationPlan with the ``indices`` dict populated
    """
    if mean_sign_filter not in (None, "positive", "negative"):
        raise ValueError(
            "mean_sign_filter must be one of: None, 'positive', 'negative'"
        )
    use_sign_filter = mean_sign_filter in ("positive", "negative")
    plan: Dict[int, Dict[str, Dict[int, List[int]]]] = {}
    allowed_layers: Optional[Set[int]] = (
        {int(layer) for layer in layer_filter} if layer_filter else None
    )

    all_labels = set(alpha_attn.keys()).union(alpha_ffn.keys())

    if strict and all_labels:
        for module in modules:
            if module == "attn":
                if not alpha_attn:
                    raise ValueError(
                        "Module 'attn' requested but no attention data found in RACE results."
                    )
                missing_labels = all_labels - set(alpha_attn.keys())
                if missing_labels:
                    raise ValueError(
                        f"Module 'attn' missing data for {len(missing_labels)} labels: "
                        f"{sorted(list(missing_labels)[:10])}{'...' if len(missing_labels) > 10 else ''}."
                    )
            elif module == "ffn":
                if not alpha_ffn:
                    raise ValueError(
                        "Module 'ffn' requested but no FFN data found in RACE results."
                    )
                missing_labels = all_labels - set(alpha_ffn.keys())
                if missing_labels:
                    raise ValueError(
                        f"Module 'ffn' missing data for {len(missing_labels)} labels: "
                        f"{sorted(list(missing_labels)[:10])}{'...' if len(missing_labels) > 10 else ''}."
                    )

    if strict and allowed_layers is not None:
        for module in modules:
            alpha_dict = alpha_attn if module == "attn" else alpha_ffn
            if not alpha_dict:
                continue
            for label_idx, layer_data in alpha_dict.items():
                missing_layers = allowed_layers - set(layer_data.keys())
                if missing_layers:
                    raise ValueError(
                        f"Label {label_idx}, Module '{module}': Missing layer data for "
                        f"layers {sorted(missing_layers)}."
                    )

    for label_idx in all_labels:
        label_plan: Dict[str, Dict[int, List[int]]] = {}

        if "attn" in modules and label_idx in alpha_attn:
            label_plan["attn"] = {}
            for layer_idx, alpha_vec in alpha_attn[label_idx].items():
                if allowed_layers is not None and layer_idx not in allowed_layers:
                    continue
                if operation_mode == "keep_top":
                    selected = select_bottom_indices(alpha_vec, top_percent)
                elif use_sign_filter and mu_attn and label_idx in mu_attn:
                    mu_vec = mu_attn[label_idx].get(layer_idx)
                    if mu_vec is not None:
                        if mean_sign_filter == "negative":
                            selected = select_top_negative_indices(
                                alpha_vec, mu_vec, top_percent
                            )
                        else:
                            selected = select_top_positive_indices(
                                alpha_vec, mu_vec, top_percent
                            )
                    else:
                        selected = select_top_indices(alpha_vec, top_percent)
                else:
                    selected = select_top_indices(alpha_vec, top_percent)
                if selected:
                    label_plan["attn"][layer_idx] = selected

        if "ffn" in modules and label_idx in alpha_ffn:
            label_plan.setdefault("ffn", {})
            for layer_idx, alpha_vec in alpha_ffn[label_idx].items():
                if allowed_layers is not None and layer_idx not in allowed_layers:
                    continue
                if operation_mode == "keep_top":
                    selected = select_bottom_indices(alpha_vec, top_percent)
                elif use_sign_filter and mu_ffn and label_idx in mu_ffn:
                    mu_vec = mu_ffn[label_idx].get(layer_idx)
                    if mu_vec is not None:
                        if mean_sign_filter == "negative":
                            selected = select_top_negative_indices(
                                alpha_vec, mu_vec, top_percent
                            )
                        else:
                            selected = select_top_positive_indices(
                                alpha_vec, mu_vec, top_percent
                            )
                    else:
                        selected = select_top_indices(alpha_vec, top_percent)
                else:
                    selected = select_top_indices(alpha_vec, top_percent)
                if selected:
                    label_plan["ffn"][layer_idx] = selected

        plan[label_idx] = label_plan

    return AblationPlan(
        indices=plan,
        modules=modules,
        top_percent=top_percent,
        operation_mode=operation_mode,
        enhancement_factor=enhancement_factor,
        layers=list(layer_filter) if layer_filter else None,
    )


# ---------------------------------------------------------------------------
# LLM plan builders
# ---------------------------------------------------------------------------


def _normalise_llm_modules(modules: Optional[List[str]]) -> Set[str]:
    """Normalize user module names to LLM ablation buckets."""
    requested = set(["attn", "mlp"] if modules is None else modules)
    normalized: Set[str] = set()
    if requested & {"attn", "attn_pre_output"}:
        normalized.add("attn")
    if requested & {"mlp", "ffn", "mlp_pre_down"}:
        normalized.add("mlp")
    return normalized


def _resolve_llm_layer_k(
    n: int,
    top_k_percent: float,
    top_k_count: Optional[int],
) -> int:
    """How many neurons to take from a pool of size ``n``."""
    if top_k_count is not None:
        k = int(top_k_count)
        if k <= 0:
            return 0
        return min(k, n)
    if top_k_percent <= 0:
        return 0
    k = max(1, int(np.ceil(n * top_k_percent / 100.0)))
    return min(k, n)


def _top_k_neuron_indices_llm(
    scores: np.ndarray,
    posterior_mean: Optional[np.ndarray],
    top_k_percent: float,
    *,
    metric: str,
    lcb_min_score: Optional[float],
    top_k_count: Optional[int] = None,
) -> Set[int]:
    """Pick top-k-percent/count neurons from one layer-local universe."""
    ranked = _ranked_neuron_indices_llm(
        scores,
        posterior_mean,
        metric=metric,
        lcb_min_score=lcb_min_score,
    )
    dim = int(np.asarray(scores).size)
    k = _resolve_llm_layer_k(dim, top_k_percent, top_k_count)
    if k <= 0:
        return set()
    return set(ranked[:k])


def _ranked_neuron_indices_llm(
    scores: np.ndarray,
    posterior_mean: Optional[np.ndarray],
    *,
    metric: str,
    lcb_min_score: Optional[float],
) -> List[int]:
    """Return a deterministic, descending layer-local neuron ranking.

    The paper's positive and negative CAM definitions assign zero, rather than
    removing a neuron from the universe, when its posterior mean has the wrong
    sign.  Applying the sign mask to the score preserves that full universe.
    ``lcb_min_score`` is an optional non-paper candidate-pool restriction.
    Ties are resolved by ascending neuron index for reproducibility.
    """
    effective_scores = np.asarray(scores, dtype=np.float64).reshape(-1).copy()
    if effective_scores.size == 0:
        return []

    if metric in ("lcb_pos", "lcb_neg"):
        if posterior_mean is None:
            raise ValueError(f"posterior_mean is required for metric {metric!r}")
        mu = np.asarray(posterior_mean, dtype=np.float64).reshape(-1)
        if mu.shape != effective_scores.shape:
            raise ValueError(
                "posterior_mean shape mismatch for sign-specific CAM metric: "
                f"scores={effective_scores.shape}, posterior_mean={mu.shape}"
            )
        sign_mask = mu > 0 if metric == "lcb_pos" else mu < 0
        effective_scores[~sign_mask] = 0.0

    candidates = np.arange(effective_scores.size, dtype=np.int64)
    if metric in ("lcb", "lcb_pos", "lcb_neg") and lcb_min_score is not None:
        candidates = candidates[effective_scores > lcb_min_score]
    if candidates.size == 0:
        return []

    # lexsort uses the last key as primary: score descending, index ascending.
    order = np.lexsort((candidates, -effective_scores[candidates]))
    return [int(index) for index in candidates[order]]


def list_llm_h5_layer_indices(
    h5_path: str,
    concept_name: str,
    modules: Optional[List[str]] = None,
) -> Set[int]:
    """Decoder layer indices present in the H5 for the given concept and modules.

    Used to size layer-by-layer jobs when some layers may have empty neuron
    selections (e.g. LCB/CAM threshold filters out all neurons).
    """
    requested_modules = _normalise_llm_modules(modules)
    out: Set[int] = set()
    from race.core.results_loader import RaceResultsReader

    with RaceResultsReader(h5_path) as reader:
        try:
            axis = reader.resolve_axis_value(concept_name, reports_only=True)
        except KeyError as exc:
            raise ValueError(str(exc)) from exc
        if "attn" in requested_modules:
            out.update(reader.list_layers(axis, "attn"))
        if "mlp" in requested_modules:
            out.update(reader.list_layers(axis, "mlp"))

    return out


def load_top_k_neurons_from_h5(
    h5_path: str,
    concept_name: str,
    top_k_percent: float,
    metric: str = "posterior_mean",
    modules: Optional[List[str]] = None,
    layer_filter: Optional[List[int]] = None,
    lcb_min_score: Optional[float] = None,
    top_k_count: Optional[int] = None,
) -> Tuple[Dict[int, Set[int]], Dict[int, Set[int]]]:
    """Load top-k% important neurons from RACE results H5 file (LLM pipeline).

    Args:
        h5_path: Path to RACE results H5 file (unified schema)
        concept_name: Concept name (e.g., "code_generation")
        top_k_percent: Percentage of neurons to select (ignored if ``top_k_count`` is set)
        metric: Metric for ranking ("posterior_mean", "lcb", "lcb_pos", "lcb_neg")
        modules: Modules to include ("attn", "mlp"). If None, use all.
        layer_filter: Layer indices to include. If None, use all.
        lcb_min_score: When metric is in the LCB family (``lcb``, ``lcb_pos``,
            ``lcb_neg``), only neurons with score **strictly greater** than this
            value are candidates;
            ``top_k_percent`` still defines k from the full layer-local
            universe. ``None`` implements the paper protocol. Setting a
            threshold is an optional extension that can return fewer than k.
        top_k_count: If not ``None``, select this many top neurons per layer (per
            attn/mlp group), capped by pool size; ``top_k_percent`` is unused for k.

    Returns:
        Tuple of (attn_neurons, mlp_neurons) dictionaries
    """
    requested_modules = _normalise_llm_modules(modules)
    allowed_layers = {int(layer) for layer in layer_filter} if layer_filter else None
    attn_neurons: Dict[int, Set[int]] = {}
    mlp_neurons: Dict[int, Set[int]] = {}
    from race.core.results_loader import RaceResultsReader

    with RaceResultsReader(h5_path) as reader:
        try:
            axis = reader.resolve_axis_value(concept_name, reports_only=True)
        except KeyError as exc:
            raise ValueError(str(exc)) from exc

        for module_name, target in (("attn", attn_neurons), ("mlp", mlp_neurons)):
            if module_name not in requested_modules:
                continue
            for layer_idx in reader.list_layers(axis, module_name):
                if allowed_layers is not None and layer_idx not in allowed_layers:
                    continue
                try:
                    scores = reader.read_metric(
                        axis,
                        module_name,
                        layer_idx,
                        metric,
                        dtype=np.float64,
                    )
                except KeyError as exc:
                    raise ValueError(str(exc)) from exc
                try:
                    mu = reader.read_metric(
                        axis,
                        module_name,
                        layer_idx,
                        "posterior_mean",
                        dtype=np.float64,
                    )
                except KeyError:
                    mu = None
                picked = _top_k_neuron_indices_llm(
                    scores,
                    mu,
                    top_k_percent,
                    metric=metric,
                    lcb_min_score=lcb_min_score,
                    top_k_count=top_k_count,
                )
                if picked:
                    target[layer_idx] = picked

    return attn_neurons, mlp_neurons


def create_ablation_plan(
    h5_path: str,
    concept_name: str,
    operation_mode: OperationMode,
    top_k_percent: float,
    metric: str = "posterior_mean",
    enhancement_factor: float = 2.0,
    modules: Optional[List[str]] = None,
    layer_filter: Optional[List[int]] = None,
    lcb_min_score: Optional[float] = None,
    top_k_count: Optional[int] = None,
) -> AblationPlan:
    """Create an ablation plan from RACE results (LLM pipeline).

    Args:
        h5_path: Path to RACE results H5 file
        concept_name: Concept name
        operation_mode: SUPPRESS or ENHANCE
        top_k_percent: Percentage of top neurons to modify
        metric: Ranking metric ("posterior_mean", "lcb", "lcb_pos", "lcb_neg")
        enhancement_factor: Amplification factor for ENHANCE mode
        modules: Modules to include ("attn", "mlp"). If None, use all.
        layer_filter: Layer indices to include. If None, use all.
        lcb_min_score: See :func:`load_top_k_neurons_from_h5`.
        top_k_count: See :func:`load_top_k_neurons_from_h5`.

    Returns:
        AblationPlan with attn_neurons/mlp_neurons populated
    """
    attn_neurons, mlp_neurons = load_top_k_neurons_from_h5(
        h5_path=h5_path,
        concept_name=concept_name,
        top_k_percent=top_k_percent,
        metric=metric,
        modules=modules,
        layer_filter=layer_filter,
        lcb_min_score=lcb_min_score,
        top_k_count=top_k_count,
    )

    return AblationPlan(
        concept_name=concept_name,
        operation_mode=(
            operation_mode.value
            if isinstance(operation_mode, OperationMode)
            else operation_mode
        ),
        top_percent=top_k_percent,
        top_k_count=top_k_count,
        attn_neurons=attn_neurons,
        mlp_neurons=mlp_neurons,
        enhancement_factor=enhancement_factor,
    )


def create_reference_filtered_ablation_plan(
    target_h5_path: str,
    target_concept: str,
    reference_h5_path: str,
    reference_concept: str,
    operation_mode: OperationMode,
    top_k_percent: float,
    metric: str = "lcb_pos",
    enhancement_factor: float = 2.0,
    modules: Optional[List[str]] = None,
    layer_filter: Optional[List[int]] = None,
    lcb_min_score: Optional[float] = None,
    top_k_count: Optional[int] = None,
    reference_top_k_percent: Optional[float] = None,
    reference_top_k_count: Optional[int] = None,
) -> AblationPlan:
    """Construct the paper's layer-local Reference-Set Filtering plan.

    For every requested layer and module this implements Eq. (RSF) directly:

    1. rank the full target neuron universe by ``metric``;
    2. form the reference exclusion set from its top ``M`` neurons using the
       same metric;
    3. traverse the target ranking, skip the exclusion set, and take the first
       ``k`` remaining neurons.

    The reference budget defaults to the target budget.  A distinct ``M`` can
    be supplied through ``reference_top_k_count`` or
    ``reference_top_k_percent``.  ``lcb_min_score`` is retained as an optional
    extension, but the paper protocol leaves it unset so rankings cover all of
    the layer-local universe.
    """
    if top_k_count is not None and top_k_count < 0:
        raise ValueError("top_k_count must be non-negative")
    if reference_top_k_count is not None and reference_top_k_count < 0:
        raise ValueError("reference_top_k_count must be non-negative")
    if reference_top_k_percent is not None and reference_top_k_percent < 0:
        raise ValueError("reference_top_k_percent must be non-negative")
    if reference_top_k_count is not None and reference_top_k_percent is not None:
        raise ValueError(
            "reference_top_k_count and reference_top_k_percent are mutually exclusive"
        )

    requested_modules = _normalise_llm_modules(modules)
    allowed_layers = {int(layer) for layer in layer_filter} if layer_filter else None
    attn_neurons: Dict[int, Set[int]] = {}
    mlp_neurons: Dict[int, Set[int]] = {}

    if reference_top_k_count is not None:
        ref_count = reference_top_k_count
        ref_percent = 0.0
    elif reference_top_k_percent is not None:
        ref_count = None
        ref_percent = reference_top_k_percent
    else:
        ref_count = top_k_count
        ref_percent = top_k_percent

    from race.core.results_loader import RaceResultsReader

    with (
        RaceResultsReader(target_h5_path) as target_reader,
        RaceResultsReader(reference_h5_path) as reference_reader,
    ):
        try:
            target_axis = target_reader.resolve_axis_value(
                target_concept, reports_only=True
            )
            reference_axis = reference_reader.resolve_axis_value(
                reference_concept, reports_only=True
            )
        except KeyError as exc:
            raise ValueError(str(exc)) from exc

        for module_name, destination in (
            ("attn", attn_neurons),
            ("mlp", mlp_neurons),
        ):
            if module_name not in requested_modules:
                continue
            reference_layers = set(
                reference_reader.list_layers(reference_axis, module_name)
            )
            for layer_idx in target_reader.list_layers(target_axis, module_name):
                if allowed_layers is not None and layer_idx not in allowed_layers:
                    continue
                if layer_idx not in reference_layers:
                    raise ValueError(
                        f"Reference H5 has no {module_name} layer {layer_idx}"
                    )

                try:
                    target_scores = target_reader.read_metric(
                        target_axis, module_name, layer_idx, metric, dtype=np.float64
                    )
                    reference_scores = reference_reader.read_metric(
                        reference_axis, module_name, layer_idx, metric, dtype=np.float64
                    )
                except KeyError as exc:
                    raise ValueError(str(exc)) from exc

                def _posterior_mean(reader, axis):
                    try:
                        return reader.read_metric(
                            axis,
                            module_name,
                            layer_idx,
                            "posterior_mean",
                            dtype=np.float64,
                        )
                    except KeyError:
                        return None

                target_mean = _posterior_mean(target_reader, target_axis)
                reference_mean = _posterior_mean(reference_reader, reference_axis)
                target_dim = int(np.asarray(target_scores).size)
                reference_dim = int(np.asarray(reference_scores).size)
                if target_dim != reference_dim:
                    raise ValueError(
                        f"RSF neuron-universe mismatch for {module_name} layer "
                        f"{layer_idx}: target={target_dim}, reference={reference_dim}"
                    )

                target_ranking = _ranked_neuron_indices_llm(
                    target_scores,
                    target_mean,
                    metric=metric,
                    lcb_min_score=lcb_min_score,
                )
                reference_ranking = _ranked_neuron_indices_llm(
                    reference_scores,
                    reference_mean,
                    metric=metric,
                    lcb_min_score=lcb_min_score,
                )
                k = _resolve_llm_layer_k(target_dim, top_k_percent, top_k_count)
                reference_k = _resolve_llm_layer_k(
                    reference_dim, ref_percent, ref_count
                )
                exclusion = set(reference_ranking[:reference_k])
                selected = [
                    index for index in target_ranking if index not in exclusion
                ][:k]
                if selected:
                    destination[layer_idx] = set(selected)
                if len(selected) < k:
                    logger.warning(
                        "RSF %s layer %d selected %d/%d neurons: only %d "
                        "ranked candidates remain outside the reference Top-%d",
                        module_name,
                        layer_idx,
                        len(selected),
                        k,
                        len(target_ranking) - len(set(target_ranking) & exclusion),
                        reference_k,
                    )

    return AblationPlan(
        concept_name=target_concept,
        operation_mode=(
            operation_mode.value
            if isinstance(operation_mode, OperationMode)
            else operation_mode
        ),
        top_percent=top_k_percent,
        top_k_count=top_k_count,
        attn_neurons=attn_neurons,
        mlp_neurons=mlp_neurons,
        enhancement_factor=enhancement_factor,
    )


def create_random_ablation_plan(
    concept_name: str,
    operation_mode: OperationMode,
    reference_plan: AblationPlan,
    seed: int = 42,
) -> AblationPlan:
    """Create a random ablation plan as a control baseline (LLM pipeline).

    Randomly selects the same number of neurons as the reference plan.

    Args:
        concept_name: Concept name (for labeling)
        operation_mode: SUPPRESS or ENHANCE
        reference_plan: Reference plan to match neuron counts
        seed: Random seed for reproducibility

    Returns:
        AblationPlan with randomly selected neurons
    """
    rng = np.random.RandomState(seed)

    attn_neurons: Dict[int, Set[int]] = {}
    mlp_neurons: Dict[int, Set[int]] = {}

    for layer_idx, ref_indices in reference_plan.attn_neurons.items():
        k = len(ref_indices)
        random_indices = rng.choice(range(10000), size=k, replace=False)
        attn_neurons[layer_idx] = set(random_indices.tolist())

    for layer_idx, ref_indices in reference_plan.mlp_neurons.items():
        k = len(ref_indices)
        random_indices = rng.choice(range(10000), size=k, replace=False)
        mlp_neurons[layer_idx] = set(random_indices.tolist())

    return AblationPlan(
        concept_name=f"{concept_name}_random",
        operation_mode=(
            operation_mode.value
            if isinstance(operation_mode, OperationMode)
            else operation_mode
        ),
        top_percent=reference_plan.top_percent,
        top_k_count=reference_plan.top_k_count,
        attn_neurons=attn_neurons,
        mlp_neurons=mlp_neurons,
        enhancement_factor=reference_plan.enhancement_factor,
    )
