import json
import random
from types import SimpleNamespace

import pytest
import torch

from race.llm.pipelines import online_race as ob


class MiniDataset:
    def __init__(self, rows):
        self.rows = [dict(row) for row in rows]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, key):
        if isinstance(key, str):
            return [row[key] for row in self.rows]
        return self.rows[key]

    @property
    def column_names(self):
        names = []
        for row in self.rows:
            for key in row:
                if key not in names:
                    names.append(key)
        return names

    def select(self, indices):
        return MiniDataset([self.rows[i] for i in indices])

    def add_column(self, name, values):
        assert len(values) == len(self.rows)
        rows = []
        for row, value in zip(self.rows, values):
            updated = dict(row)
            updated[name] = value
            rows.append(updated)
        return MiniDataset(rows)

    def remove_columns(self, names):
        if isinstance(names, str):
            names = [names]
        names = set(names)
        return MiniDataset(
            [
                {key: value for key, value in row.items() if key not in names}
                for row in self.rows
            ]
        )

    def map(self, fn, batch_size=None):
        del batch_size
        rows = []
        for row in self.rows:
            updated = dict(row)
            mapped = fn(dict(row))
            if mapped:
                updated.update(mapped)
            rows.append(updated)
        return MiniDataset(rows)


class LengthTokenizer:
    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return text.split()


class OffsetTokenizer:
    padding_side = "left"

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return list(range(len(text)))

    def __call__(
        self,
        texts,
        *,
        return_tensors,
        padding,
        truncation,
        return_offsets_mapping=False,
    ):
        del return_tensors, padding, truncation
        max_length = max(len(text) for text in texts)
        attention_masks = []
        offsets = []
        for text in texts:
            padding_length = max_length - len(text)
            attention_masks.append([0] * padding_length + [1] * len(text))
            offsets.append(
                [(0, 0)] * padding_length
                + [(index, index + 1) for index in range(len(text))]
            )
        result = {"attention_mask": torch.tensor(attention_masks)}
        if return_offsets_mapping:
            result["offset_mapping"] = torch.tensor(offsets)
        return result


class FakeRecordResult:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.num_generated_tokens = [2] * batch_size
        self.generated_token_ids = [[1, 2]] * batch_size


class FakeRecorder:
    tokenizer = LengthTokenizer()
    device = "cpu"

    def __init__(self):
        self.record_sample_calls = []
        self.teacher_forcing_calls = []
        self.stream_calls = []

    def record_sample(self, *, prompts, sample_indices, is_correct, **gen_kwargs):
        self.record_sample_calls.append(
            {
                "prompts": prompts,
                "sample_indices": sample_indices,
                "is_correct": is_correct,
                "gen_kwargs": gen_kwargs,
            }
        )
        return FakeRecordResult(len(prompts))

    def record_sample_teacher_forcing(self, *, prompts, sample_indices, is_correct):
        self.teacher_forcing_calls.append(
            {
                "prompts": prompts,
                "sample_indices": sample_indices,
                "is_correct": is_correct,
            }
        )
        return FakeRecordResult(len(prompts))

    def stream_evidence(self, result, accumulator, domain, token_masks=None):
        self.stream_calls.append(
            {
                "batch_size": result.batch_size,
                "accumulator": accumulator,
                "domain": domain,
                "token_masks": token_masks,
            }
        )
        return result.batch_size


def _patch_dataset_spec(monkeypatch, spec):
    monkeypatch.setattr(ob, "ensure_builtin_datasets_registered", lambda: None)
    monkeypatch.setattr(ob, "get_dataset_spec", lambda dataset_id: spec)


def test_prepare_dataset_random_sample_adds_original_sample_indices(monkeypatch):
    rows = MiniDataset({"id": i} for i in range(5))
    load_calls = []

    def load_fn(**kwargs):
        load_calls.append(kwargs)
        return rows

    def format_fn(example, **kwargs):
        assert kwargs["split"] == "complete"
        assert kwargs["instruction_prefix"] == "I:"
        assert kwargs["response_prefix"] == "R:"
        assert kwargs["prefill"] is True
        assert kwargs["tokenizer"] is FakeRecorder.tokenizer
        return {"formatted_prompt": f"prompt-{example['id']}"}

    spec = SimpleNamespace(dataset_id="toy", load_fn=load_fn, format_fn=format_fn)
    _patch_dataset_spec(monkeypatch, spec)

    dataset = ob._prepare_dataset(
        FakeRecorder(),
        dataset_id="toy",
        dataset_name="name",
        dataset_split="split",
        max_samples=2,
        random_sample=True,
        sample_seed=7,
        prompt_split="complete",
        instruction_prefix="I:",
        response_prefix="R:",
        prefill=True,
    )

    expected_indices = random.Random(7).sample(range(5), 2)
    assert load_calls == [
        {"max_samples": None, "dataset_name": "name", "split": "split"}
    ]
    assert dataset["id"] == expected_indices
    assert dataset["sample_index"] == expected_indices
    assert dataset["formatted_prompt"] == [f"prompt-{i}" for i in expected_indices]


def test_prepare_dataset_delegates_bounded_sampling_to_loader(monkeypatch):
    rows = MiniDataset({"id": i} for i in range(2))
    load_calls = []

    def load_fn(**kwargs):
        load_calls.append(kwargs)
        return rows

    spec = SimpleNamespace(
        dataset_id="streaming_toy",
        load_fn=load_fn,
        format_fn=lambda example, **kwargs: {
            "formatted_prompt": f"prompt-{example['id']}"
        },
        loader_handles_sampling=True,
    )
    _patch_dataset_spec(monkeypatch, spec)

    dataset = ob._prepare_dataset(
        FakeRecorder(),
        dataset_id="streaming_toy",
        dataset_name="name",
        dataset_split="train",
        max_samples=2,
        random_sample=True,
        sample_seed=17,
        prompt_split="complete",
        instruction_prefix="",
        response_prefix="",
        prefill=False,
    )

    assert load_calls == [
        {
            "max_samples": 2,
            "random_sample": True,
            "sample_seed": 17,
            "dataset_name": "name",
            "split": "train",
        }
    ]
    assert dataset["id"] == [0, 1]


def test_load_evalscope_generations_maps_math500_level_local_indices(tmp_path):
    dataset = MiniDataset(
        [
            {"level": "1"},
            {"level": "2"},
            {"level": "1"},
        ]
    )
    (tmp_path / "math_500_Level 1.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {"index": 0, "messages": [{"role": "assistant", "content": "A"}]}
                ),
                json.dumps(
                    {"index": 1, "messages": [{"role": "assistant", "content": "B"}]}
                ),
                json.dumps(
                    {"index": 2, "messages": [{"role": "assistant", "content": "skip"}]}
                ),
                "not json",
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "math_500_Level 2.jsonl").write_text(
        json.dumps(
            {"index": 0, "model_output": {"choices": [{"message": {"content": "C"}}]}}
        ),
        encoding="utf-8",
    )

    generations = ob._load_evalscope_generations(
        str(tmp_path),
        dataset_id="math500",
        dataset=dataset,
    )

    assert generations == {0: "A", 2: "B", 1: "C"}


def test_prepare_evalscope_dataset_appends_matched_teacher_outputs(
    monkeypatch, tmp_path
):
    source = MiniDataset([{"id": 0}, {"id": 1}, {"id": 2}])

    def format_fn(example, **kwargs):
        assert kwargs["prefill"] is False
        return {"formatted_prompt": f"prompt-{example['id']}:"}

    spec = SimpleNamespace(
        dataset_id="mbpp_plus",
        load_fn=lambda **kwargs: source,
        format_fn=format_fn,
    )
    _patch_dataset_spec(monkeypatch, spec)
    (tmp_path / "mbpp_plus_outputs.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {"index": 2, "messages": [{"role": "assistant", "content": "C"}]}
                ),
                json.dumps(
                    {"index": 0, "messages": [{"role": "assistant", "content": "A"}]}
                ),
            ]
        ),
        encoding="utf-8",
    )

    dataset = ob._prepare_evalscope_dataset(
        FakeRecorder(),
        dataset_id="mbpp_plus",
        dataset_name=None,
        dataset_split=None,
        max_samples=None,
        random_sample=False,
        sample_seed=42,
        prompt_split="instruct",
        instruction_prefix="",
        response_prefix="",
        results_dir=str(tmp_path),
        evidence_scope="assistant",
    )

    assert dataset["id"] == [0, 2]
    assert dataset["formatted_prompt"] == ["prompt-0:A", "prompt-2:C"]
    assert dataset["sample_index"] == [0, 2]
    assert dataset["evalscope_evidence_start_char"] == [9, 9]


def test_prepare_evalscope_dataset_raises_when_no_records_match(monkeypatch, tmp_path):
    spec = SimpleNamespace(
        dataset_id="mbpp_plus",
        load_fn=lambda **kwargs: MiniDataset([{"id": 0}]),
        format_fn=lambda example, **kwargs: {"formatted_prompt": "prompt:"},
    )
    _patch_dataset_spec(monkeypatch, spec)
    (tmp_path / "mbpp_plus_outputs.jsonl").write_text(
        json.dumps(
            {"index": 7, "messages": [{"role": "assistant", "content": "late"}]}
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="No EvalScope samples"):
        ob._prepare_evalscope_dataset(
            FakeRecorder(),
            dataset_id="mbpp_plus",
            dataset_name=None,
            dataset_split=None,
            max_samples=None,
            random_sample=False,
            sample_seed=42,
            prompt_split="instruct",
            instruction_prefix="",
            response_prefix="",
            results_dir=str(tmp_path),
        )


def test_process_batches_sorts_by_length_and_uses_sample_index():
    recorder = FakeRecorder()
    dataset = MiniDataset(
        [
            {"formatted_prompt": "three token prompt", "sample_index": 30},
            {"formatted_prompt": "one", "sample_index": 10},
            {"formatted_prompt": "two tokens", "sample_index": 20},
        ]
    )

    stats = ob._process_batches(
        recorder,
        dataset,
        accumulator=object(),
        domain="toy",
        dataset_id="toy",
        batch_size=2,
        gen_kwargs={"max_new_tokens": 4, "temperature": 0.0},
        sort_by_length=True,
        teacher_forcing=False,
    )

    assert stats["total"] == 3
    assert stats["processed"] == 3
    assert [call["prompts"] for call in recorder.record_sample_calls] == [
        ["one", "two tokens"],
        ["three token prompt"],
    ]
    assert [call["sample_indices"] for call in recorder.record_sample_calls] == [
        [10, 20],
        [30],
    ]
    assert all(
        call["gen_kwargs"] == {"max_new_tokens": 4, "temperature": 0.0}
        for call in recorder.record_sample_calls
    )
    assert [call["domain"] for call in recorder.stream_calls] == ["toy", "toy"]
    assert [call["token_masks"] for call in recorder.stream_calls] == [None, None]


def test_process_batches_teacher_forcing_uses_prefill_recorder():
    recorder = FakeRecorder()
    dataset = MiniDataset(
        [
            {"formatted_prompt": "a"},
            {"formatted_prompt": "b"},
        ]
    )

    stats = ob._process_batches(
        recorder,
        dataset,
        accumulator=object(),
        domain="toy",
        dataset_id="toy",
        batch_size=2,
        gen_kwargs={"max_new_tokens": 4},
        sort_by_length=False,
        teacher_forcing=True,
    )

    assert stats["processed"] == 2
    assert recorder.record_sample_calls == []
    assert recorder.teacher_forcing_calls == [
        {
            "prompts": ["a", "b"],
            "sample_indices": [0, 1],
            "is_correct": [True, True],
        }
    ]


def test_process_batches_masks_teacher_forcing_to_assistant_span():
    recorder = FakeRecorder()
    recorder.tokenizer = OffsetTokenizer()
    dataset = MiniDataset(
        [
            {
                "formatted_prompt": "xxAB",
                "evalscope_evidence_start_char": 2,
            },
            {
                "formatted_prompt": "xC",
                "evalscope_evidence_start_char": 1,
            },
        ]
    )

    stats = ob._process_batches(
        recorder,
        dataset,
        accumulator=object(),
        domain="toy",
        dataset_id="toy",
        batch_size=2,
        gen_kwargs={},
        sort_by_length=False,
        teacher_forcing=True,
    )

    assert stats["processed"] == 2
    assert recorder.stream_calls[0]["token_masks"] == [
        [False, False, True, True],
        [False, False, False, True],
    ]
