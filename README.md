# RACE: Scalable Statistical Estimation of Functional Consistency in LLM Neurons

## Abstract

> Discovering stable neuron behavior across entire domains remains a challenge in mechanistic interpretability. Existing methods often rely on instance-level point estimates or computationally expensive procedures, which either obscure population-level variability or limit scalable domain-wide analysis. We present RACE (Residual Alignment for Consistency Estimation), a forward-pass statistical framework that evaluates the domain-wide functional consistency of Transformer neurons. Compared with gradient-based point estimates, RACE produces neuron rankings that yield more domain-specific effects under perturbation. Token-distribution shifts support the connection between the selected neurons and the target domain, while scoring requires roughly one-hundredth of the computational overhead of the gradient-based methods.

## Method overview

RACE processes each attention or feed-forward module in two stages:

1. Residual-Direction Alignment (RDA) projects each neuron's contribution onto
   the direction of the module's residual update. This produces signed evidence
   for every neuron and token.
2. A Normal-Inverse-Gamma posterior aggregates the evidence online. The
   posterior mean records direction and magnitude, while positive Conservative
   Alignment Magnitude (CAM) penalizes uncertain estimates.

The streaming implementation keeps sufficient statistics. See [the method reference](docs/method.md) for the
equations and [the HDF5 schema](docs/RACE_H5_schema.md) for output fields.

## Repository layout

```text
packages/race/       Core scoring, recording, dataset, and HDF5 code
packages/race-eval/  Neuron selection, intervention, and evaluation code
scripts/             Reproduction and evaluation entry points
tests/               Unit tests for the public implementation
docs/                Method, pipeline, schema, and reproduction references
```

## Requirements

- Linux
- Python 3.10 or newer
- A CUDA-capable GPU for model-scale experiments
- [`uv`](https://docs.astral.sh/uv/) for the documented environment workflow
- Access to the model and benchmark datasets used by your experiment

Model-scale runs can require substantial GPU memory. Use `--model-device-map auto` and `--accumulator-device cpu` when the model does not fit on one GPU.

## Install

Clone the repository and install the workspace:

```bash
git clone https://github.com/Nexround/RACE.git
cd RACE
uv sync --frozen
```

For development, activate the environment or prefix commands with `uv run`:

```bash
source .venv/bin/activate
python -c "import race; print(race.__version__)"
```

## Run a small LLM experiment

This command runs online RACE on ten MATH-500 examples and writes an HDF5
result under `results/quickstart/`:

```bash
uv run python -m race.llm.pipelines.online_race \
  --dataset math500 \
  --model Qwen/Qwen2.5-Coder-1.5B-Instruct \
  --output-dir results/quickstart \
  --max-samples 10 \
  --greedy \
  --no-compile
```

Successful runs print the final HDF5 path. The file contains per-layer attention
and MLP scores, posterior sufficient statistics, and run metadata.

## Create an intervened model

Use an HDF5 result to suppress the highest-scoring neurons and save a standard
Hugging Face checkpoint:

```bash
uv run python -m race_eval.llm.pipelines.save_perturbed_model \
  --model Qwen/Qwen2.5-Coder-1.5B-Instruct \
  --race-h5 results/quickstart/math500_online_race.h5 \
  --concept math500 \
  --operation suppress \
  --metric lcb_pos \
  --top-k-percent 1.0 \
  --modules attn mlp \
  --output-dir results/perturbed_models
```

Timestamped runs may place the HDF5 file one directory below the path shown
above. Use the path printed by the scoring command.

## Supported evidence modes

The LLM pipeline supports:

- autoregressive output-token evidence;
- first-generated-token evidence;
- teacher-forced evidence over fixed input sequences;
- replay of EvalScope predictions, using either the full prompt-response
  sequence or assistant-response tokens only.

See [LLM evidence strategies](docs/online_race_evidence_strategies.md) for the
semantics and constraints of each mode.

## Dataset

The PyComp-1K dataset referenced in the paper is available on Hugging Face:
[nexround/PyComp-1K](https://huggingface.co/datasets/nexround/PyComp-1K).

## License

RACE is released under the [MIT License](LICENSE). Dataset and model licenses
remain with their respective owners.
