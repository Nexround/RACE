"""Post-hoc recomputation of RACE posteriors with a new gamma parameter.

Given an existing RACE H5 results file, this script reconstructs the NIG
posterior from the saved sufficient statistics (``nig_sum_e``, ``nig_sum_e2``,
and ``N``) and recomputes ``cam_score`` under a new ``gamma`` — **without
re-running model inference**.

Reversibility summary
---------------------
=========  ==========  ================================================
Parameter  Reversible  Reason
=========  ==========  ================================================
gamma      YES         Only used for the final CAM-score t-quantile step
mu_0       NO          Changes the prior mean; baked into μ_n
lambda_0   NO          Affects λ_n weighting; baked into sufficient stats
alpha_0    NO          Baked into α_n
beta_0     NO          Baked into β_n
=========  ==========  ================================================

The script reads from the source H5, restores each ``NIGPosterior`` from its
sufficient statistics, recomputes ``cam_score`` with the new ``gamma``, and
writes an updated H5 file.

Usage::

    uv run python -m race.core.recompute_posteriors \\
        --input results/20250101_120000/cifar100_online_race.h5 \\
        --output results/recomputed.h5 \\
        --gamma 0.01

Or programmatically::

    from race.core.recompute_posteriors import recompute_posteriors

    recompute_posteriors(
        input_path="results/original.h5",
        output_path="results/recomputed.h5",
        new_gamma=0.01,
    )
"""

from __future__ import annotations

import argparse
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import h5py
import numpy as np

from race.core.analyzer import NIGPosterior

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Layer-level recomputation
# ---------------------------------------------------------------------------


def _recompute_layer(
    layer_group: h5py.Group,
    new_gamma: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Recompute cam_score for a single layer group using a new gamma.

    The layer group must contain:
    - Dataset ``nig_sum_e``  (float64, shape [dim])
    - Dataset ``nig_sum_e2`` (float64, shape [dim])
    - Attr    ``N``          (int)
    - Attr    ``mu_0``       (float)
    - Attr    ``lambda_0``   (float)
    - Attr    ``alpha_0``    (float)
    - Attr    ``beta_0``     (float)

    Returns:
        ``(cam_score, negative_cam_score)`` as float32 arrays.
    """
    if "nig_sum_e" not in layer_group or "nig_sum_e2" not in layer_group:
        raise ValueError(
            f"Layer group '{layer_group.name}' is missing 'nig_sum_e' / 'nig_sum_e2'. "
            "This file does not contain the sufficient statistics required "
            "for posterior recomputation."
        )

    sum_e = layer_group["nig_sum_e"][()].astype(np.float64)
    sum_e2 = layer_group["nig_sum_e2"][()].astype(np.float64)
    N = int(layer_group.attrs.get("N", 0))
    mu_0 = float(layer_group.attrs.get("mu_0", 0.0))
    lambda_0 = float(layer_group.attrs.get("lambda_0", 1.0))
    alpha_0 = float(layer_group.attrs.get("alpha_0", 1.0))
    beta_0 = float(layer_group.attrs.get("beta_0", 1.0))

    dim = len(sum_e)
    import torch

    nig = NIGPosterior.from_state_dict(
        {
            "mu_0": mu_0,
            "lambda_0": lambda_0,
            "alpha_0": alpha_0,
            "beta_0": beta_0,
            "N": N,
            "sum_e": sum_e,
            "sum_e2": sum_e2,
        },
        dim=dim,
    )

    cam = nig.cam_score(gamma=new_gamma).cpu().numpy().astype(np.float32)
    neg_cam = nig.negative_cam_score(gamma=new_gamma).cpu().numpy().astype(np.float32)
    return cam, neg_cam


# ---------------------------------------------------------------------------
# File-level recomputation
# ---------------------------------------------------------------------------


def recompute_posteriors(
    input_path: str,
    output_path: str,
    new_gamma: Optional[float] = None,
) -> str:
    """Recompute RACE posteriors from an existing H5 file with a new gamma.

    Args:
        input_path: Path to the original RACE H5 results file.
        output_path: Path for the new H5 file with recomputed cam_score.
        new_gamma: New gamma for CAM-score (``None`` = keep original).

    Returns:
        The resolved output path.
    """
    input_path = str(Path(input_path).resolve())
    output_path = str(Path(output_path).resolve())

    if input_path == output_path:
        raise ValueError("Input and output paths must differ to avoid data loss.")

    # Read the gamma stored in the source file.
    with h5py.File(input_path, "r") as h5f:
        orig_gamma = None
        cfg_group = h5f.get("meta/race_config")
        if cfg_group is not None:
            orig_gamma = float(cfg_group.attrs.get("gamma", 0.05))
        metrics = h5f.get("reports/summary/metrics")
        if metrics is not None and orig_gamma is None:
            orig_gamma = float(metrics.attrs.get("gamma", 0.05))

    if orig_gamma is None:
        orig_gamma = 0.05
    eff_gamma = new_gamma if new_gamma is not None else orig_gamma

    logger.info("=" * 70)
    logger.info("  RACE Posterior Recomputation")
    logger.info("=" * 70)
    logger.info("  Input        : %s", input_path)
    logger.info("  Output       : %s", output_path)
    logger.info("  Original gamma : %s", orig_gamma)
    logger.info("  New gamma      : %s", eff_gamma)
    logger.info("=" * 70)

    # -- Copy H5 and modify cam_score in-place on the copy --
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    logger.info("  Copying source H5 (%s) ...", input_path)
    shutil.copy2(input_path, output_path)
    logger.info("  Copy complete.  Recomputing cam_score ...")

    layers_updated = 0
    domains_updated = 0

    with h5py.File(output_path, "a") as h5f:
        # -- Update config metadata --
        for meta_key in ("meta/race_config",):
            cfg_grp = h5f.get(meta_key)
            if cfg_grp is not None:
                cfg_grp.attrs["gamma"] = eff_gamma

        # -- Add recompute provenance --
        recompute_meta = h5f.require_group("meta/recompute_history")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        entry = recompute_meta.require_group(ts)
        entry.attrs["source_file"] = input_path
        entry.attrs["timestamp"] = ts
        entry.attrs["original_gamma"] = orig_gamma
        entry.attrs["new_gamma"] = eff_gamma

        # -- Iterate over all domain reports --
        reports_axes = h5f.get("reports/axes")
        if reports_axes is None:
            raise ValueError("H5 file does not contain 'reports/axes'")

        domain_keys = sorted(k for k in reports_axes.keys() if k.startswith("value_"))
        logger.info("  Processing %d domains ...", len(domain_keys))

        try:
            from tqdm import tqdm

            domain_iter = tqdm(
                domain_keys, desc="  Recomputing", unit="domain", ncols=100
            )
        except ImportError:
            domain_iter = domain_keys

        for value_name in domain_iter:
            domain_group = reports_axes[value_name]
            modules_group = domain_group.get("modules")
            if modules_group is None:
                continue

            for module_name in list(modules_group.keys()):
                module_group = modules_group[module_name]
                for layer_key in sorted(
                    k for k in module_group.keys() if k.startswith("layer_")
                ):
                    layer_group = module_group[layer_key]

                    if "nig_sum_e" not in layer_group:
                        continue

                    new_cam, new_neg_cam = _recompute_layer(layer_group, eff_gamma)

                    # Overwrite cam_score dataset
                    if "cam_score" in layer_group:
                        del layer_group["cam_score"]
                    layer_group.create_dataset(
                        "cam_score", data=new_cam, compression="gzip"
                    )
                    if "negative_cam_score" in layer_group:
                        del layer_group["negative_cam_score"]
                    layer_group.create_dataset(
                        "negative_cam_score",
                        data=new_neg_cam,
                        compression="gzip",
                    )

                    layers_updated += 1

            domains_updated += 1

        # -- Update summary metrics --
        summary_metrics = h5f.get("reports/summary/metrics")
        if summary_metrics is not None:
            summary_metrics.attrs["gamma"] = eff_gamma

    logger.info("  Recomputation complete!")
    logger.info("  Domains updated : %d", domains_updated)
    logger.info("  Layers updated  : %d", layers_updated)
    logger.info("  Output          : %s", output_path)
    logger.info("=" * 70)

    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Recompute RACE cam_score from an existing H5 file with a new gamma "
            "(no re-inference needed)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m race.core.recompute_posteriors \\\n"
            "      --input results/original.h5 \\\n"
            "      --output results/recomputed.h5 \\\n"
            "      --gamma 0.01\n"
        ),
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Path to the original RACE H5 results file",
    )
    parser.add_argument(
        "--output", "-o", type=str, required=True, help="Path for the new H5 file"
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=None,
        help="New gamma for CAM-score confidence level (default: keep original)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.gamma is None:
        parser.error("At least --gamma must be specified.")

    recompute_posteriors(
        input_path=args.input,
        output_path=args.output,
        new_gamma=args.gamma,
    )


if __name__ == "__main__":
    main()
