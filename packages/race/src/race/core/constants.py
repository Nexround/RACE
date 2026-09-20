"""Shared constants for the RACE package.

Centralizes magic strings and lookup tables that are used across multiple
modules so that they are defined in exactly one place.
"""

from __future__ import annotations

from typing import Dict

# ---------------------------------------------------------------------------
# Module name aliases
# ---------------------------------------------------------------------------
# Maps the raw HDF5 module group names written by recorders to the short
# canonical names used throughout analysis, ablation plans, and reports.

MODULE_ALIASES: Dict[str, str] = {
    "attn_pre_output": "attn",
    "mlp_pre_down": "ffn",
}

# Supported dataset names for the per-neuron confidence score.
LCB_DATASET_ALIASES = {"lcb", "credible_lcb", "cam_score"}
