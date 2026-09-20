# RACE method reference

This page defines the computation implemented in `race.core.analyzer` and
`race.core.online_accumulator`. It is a reference for readers verifying the
paper implementation.

## Residual-Direction Alignment

For a module residual update `delta_r`, RACE defines a self-bootstrapped unit
direction:

```text
d_hat = delta_r / (||delta_r||_2 + epsilon)
```

For an intermediate neuron activation `a_j` and the associated output-projection
vector `w_j`, the signed evidence is:

```text
e_j = a_j * (w_j^T d_hat)
```

The attention implementation applies this operation to the input of the output
projection. The MLP implementation applies it to the input of the down
projection. Positive and negative values retain their direction during
aggregation.

## Normal-Inverse-Gamma aggregation

For each neuron, RACE models evidence as `e_j ~ Normal(mu_j, sigma_j^2)` with a
Normal-Inverse-Gamma prior `(mu_0, lambda_0, alpha_0, beta_0)`. The online state
stores only `N`, `sum(e)`, and `sum(e^2)`.

Given the empirical mean `e_bar` and centered sum of squares `S`, the posterior
parameters are:

```text
lambda_n = lambda_0 + N
alpha_n  = alpha_0 + N / 2
mu_n     = (lambda_0 * mu_0 + N * e_bar) / lambda_n
beta_n   = beta_0 + S / 2
           + lambda_0 * N * (e_bar - mu_0)^2 / (2 * lambda_n)
```

The default prior is defined by `OnlineRACEConfig`. Pass an explicit config when
reproducing a run so that prior changes are visible in the run metadata.

## Conservative Alignment Magnitude

RACE reports the signed posterior mean and the paper's positive Conservative
Alignment Magnitude (CAM):

```text
sigma_mu^2 = beta_n / (alpha_n * lambda_n)
rho_j      = mu_n,j - t_(1-gamma, 2*alpha_n) * sigma_mu,j
CAM_j      = I[mu_n,j > 0] * max(0, rho_j)
```

The negative-direction ablation is stored separately:

```text
NegCAM_j = I[mu_n,j < 0]
           * max(0, -mu_n,j - t_(1-gamma, 2*alpha_n) * sigma_mu,j)
```

Use `lcb_pos` for the paper's default CAM ranking and `lcb_neg` only for the
negative-CAM ablation.

## Reference-Set Filtering

Reference-Set Filtering (RSF) is applied independently for each layer and
module. Let `U^ell` be that layer-local neuron universe, let `B_ref^ell` be the
reference Top-M set, and let `Pi_tar^ell` be the complete target ranking under
the same scoring rule. The implementation computes:

```text
Pi_RSF^ell = [j in Pi_tar^ell where j not in B_ref^ell]
S_spec^ell = First_k(Pi_RSF^ell)
```

The selector scans past excluded target neurons instead of subtracting the
reference set from an already truncated target Top-k. It therefore returns
exactly `k` neurons whenever at least `k` members of `U^ell` remain outside the
reference Top-M set. Target and reference rankings always use the same metric.
The reference budget defaults to the target budget, but `--general-top-k` or
`--general-top-k-percent` can set `M` independently.

Do not pass `--lcb-min-score` for the paper protocol. That option restricts the
candidate universe and is retained only for non-paper threshold experiments.

## Streaming behavior

`OnlineRACEAccumulator` updates each posterior as activations arrive, then lets
the recorder release the activation tensors. The HDF5 output retains posterior
parameters, derived scores, axis metadata, sample counts, and run configuration.
It does not retain the full activation stream.

## Source map

| Method component | Implementation |
| --- | --- |
| RDA evidence and NIG posterior | `packages/race/src/race/core/analyzer.py` |
| Streaming aggregation | `packages/race/src/race/core/online_accumulator.py` |
| LLM activation hooks | `packages/race/src/race/llm/recording/activation.py` |
| LLM experiment pipeline | `packages/race/src/race/llm/pipelines/online_race.py` |
| HDF5 schema | `packages/race/src/race/io/h5_schema.py` |
| Neuron selection | `packages/race-eval/src/race_eval/ablation/plan.py` |
| Checkpoint intervention | `packages/race-eval/src/race_eval/llm/pipelines/save_perturbed_model.py` |
