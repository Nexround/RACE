from types import SimpleNamespace

import pytest
from datasets import Dataset

from race.llm.dataset import mmlu_redux


def _subject_dataset(question: str = "Question?") -> Dataset:
    return Dataset.from_dict(
        {
            "question": [question],
            "choices": [["one", "two", "three", "four"]],
            "answer": [1],
        }
    )


def test_load_mmlu_redux_defaults_to_all_57_subjects(monkeypatch):
    loaded_subjects = []

    def fake_load_subject(dataset_name, subject, split):
        assert dataset_name == mmlu_redux._MODELSCOPE_DATASET
        assert split == "test"
        loaded_subjects.append(subject)
        return _subject_dataset(subject)

    monkeypatch.setattr(mmlu_redux, "_load_subject", fake_load_subject)

    dataset = mmlu_redux.load_mmlu_redux()

    assert loaded_subjects == mmlu_redux._ALL_SUBJECTS
    assert len(dataset) == 5700 // 100
    assert dataset["subject"] == mmlu_redux._ALL_SUBJECTS


def test_load_mmlu_redux_rejects_partial_full_dataset(monkeypatch):
    def fake_load_subject(dataset_name, subject, split):
        del dataset_name, split
        if subject == "missing":
            raise FileNotFoundError("not cached")
        return _subject_dataset(subject)

    monkeypatch.setattr(mmlu_redux, "_load_subject", fake_load_subject)

    with pytest.raises(RuntimeError, match="Failed to load 1/2"):
        mmlu_redux.load_mmlu_redux(subjects=["present", "missing"])


def test_format_prompt_matches_evalscope_multiple_choice_template():
    tokenizer = SimpleNamespace(chat_template=None)
    example = {
        "question": "What is two plus two?",
        "choices": ["1", "2", "3", "4"],
    }

    result = mmlu_redux.format_prompt_for_model(example, tokenizer)

    assert result["formatted_prompt"] == (
        "Answer the following multiple choice question. The last line of your "
        "response should be of the following format: 'ANSWER: [LETTER]' "
        "(without quotes) where [LETTER] is one of A,B,C,D. Think step by step "
        "before answering.\n\nWhat is two plus two?\n\n"
        "A) 1\nB) 2\nC) 3\nD) 4\n"
    )
