# Add a Dataset to Online RACE

This guide is for implementers adding a dataset adapter to the unified
`race.llm.pipelines.online_race` pipeline. The `--dataset <id>` option selects a
registered adapter.

## Implement the adapter contract

Create a module in `race.llm.dataset` that exposes these three components:

- `defaults`: a `dict[str, Any]` with at least `max_new_tokens`, `temperature`,
  `top_p`, `instruction_prefix`, and `response_prefix`.
- `load_fn`: a function with a `load_<dataset>(...) -> datasets.Dataset`
  signature. It must return a Hugging Face `Dataset`, not a `DatasetDict`. It
  should add a common `prompt` field and support `max_samples` for smoke tests.
- `format_fn`: a function with a
  `format_prompt_for_model(example, tokenizer, ...) -> dict` signature. It must
  add a `formatted_prompt` string for each example. It should use
  `apply_chat_template` when the tokenizer defines a chat template and use a
  plain string otherwise.

See `packages/race/src/race/llm/dataset/math500.py` for an existing adapter.

## Create the dataset module

For a dataset named `mydataset`, create
`packages/race/src/race/llm/dataset/mydataset.py`:

```python
from __future__ import annotations

from typing import Any

from datasets import Dataset, load_dataset

MYDATASET_DEFAULTS: dict[str, Any] = {
    "max_new_tokens": 256,
    "temperature": 0.0,
    "top_p": 1.0,
    "instruction_prefix": "Answer the question:",
    "response_prefix": "Answer:",
}


def load_mydataset(
    max_samples: int | None = None,
    dataset_name: str = "org/mydataset",
    split: str | None = None,
) -> Dataset:
    dataset = load_dataset(dataset_name, split=split or "train")
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    def add_prompt(example: dict[str, Any]) -> dict[str, Any]:
        example["prompt"] = example["question"]
        return example

    return dataset.map(add_prompt)


def format_prompt_for_model(
    example: dict[str, Any],
    tokenizer: Any,
    split: str = "instruct",
    instruction_prefix: str = MYDATASET_DEFAULTS["instruction_prefix"],
    response_prefix: str = MYDATASET_DEFAULTS["response_prefix"],
    prefill: bool = True,
    direct_completion: bool = False,
) -> dict[str, Any]:
    del split, prefill
    prompt = example["prompt"]
    if tokenizer.chat_template is None or direct_completion:
        example["formatted_prompt"] = (
            f"{instruction_prefix}\n{prompt}\n{response_prefix}\n"
        )
        return example

    example["formatted_prompt"] = tokenizer.apply_chat_template(
        [{"role": "user", "content": f"{instruction_prefix}\n{prompt}"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    return example
```

## Register the adapter

Register the adapter in `ensure_builtin_datasets_registered()` in
`packages/race/src/race/llm/dataset/registry.py`:

```python
from race.llm.dataset.mydataset import (
    MYDATASET_DEFAULTS,
    format_prompt_for_model as mydataset_format,
    load_mydataset,
)

register_dataset(
    DatasetSpec(
        dataset_id="mydataset",
        defaults=MYDATASET_DEFAULTS,
        load_fn=load_mydataset,
        format_fn=mydataset_format,
    )
)
```

If the `--dataset` argument in `online_race.py` uses explicit choices, add
`mydataset` to those choices.

## Verify the adapter

Run one sample with first-token evidence:

```bash
uv run python -m race.llm.pipelines.online_race \
  --dataset mydataset \
  --max-samples 1 \
  --first-token-only
```

The run should load one example, create its `formatted_prompt`, and begin model
inference without a registry or schema error.

For a lightweight unit test, call the loader and formatter directly with a
small fixture or mocked dataset. Do not add temporary validation code to the
adapter module.

## Common integration errors

- Missing `formatted_prompt`: the pipeline reads
  `dataset["formatted_prompt"]` when it creates batches.
- Returning `DatasetDict`: select a split explicitly so the loader returns a
  `Dataset`.
- Assuming every tokenizer has a chat template: use plain-string formatting
  when `tokenizer.chat_template is None`.
