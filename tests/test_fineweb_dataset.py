import pytest

from race.llm.dataset import fineweb


class FakeStream:
    def __init__(self, rows):
        self.rows = list(rows)
        self.shuffle_calls = []

    def __iter__(self):
        return iter(self.rows)

    def shuffle(self, *, seed, buffer_size):
        self.shuffle_calls.append({"seed": seed, "buffer_size": buffer_size})
        return self


def test_load_fineweb_streams_shuffles_filters_and_truncates(monkeypatch):
    stream = FakeStream(
        [
            {"text": "short", "id": "skip"},
            {
                "text": "alpha beta gamma delta",
                "id": "doc-1",
                "dump": "dump-1",
                "url": "https://example.com/1",
                "language_score": 0.99,
                "token_count": 4,
            },
            {
                "text": "one two three four five",
                "id": "doc-2",
                "dump": "dump-2",
                "url": "https://example.com/2",
                "language_score": 0.98,
                "token_count": 5,
            },
        ]
    )
    load_calls = []

    def fake_load_dataset(*args, **kwargs):
        load_calls.append((args, kwargs))
        return stream

    monkeypatch.setattr(fineweb, "load_dataset", fake_load_dataset)

    dataset = fineweb.load_fineweb(
        max_samples=2,
        random_sample=True,
        sample_seed=7,
        shuffle_buffer_size=11,
        min_chars=10,
        max_chars=20,
    )

    assert load_calls == [
        (
            ("HuggingFaceFW/fineweb",),
            {"name": "sample-10BT", "split": "train", "streaming": True},
        )
    ]
    assert stream.shuffle_calls == [{"seed": 7, "buffer_size": 11}]
    assert dataset["text"] == ["alpha beta gamma", "one two three four"]
    assert dataset["sample_index"] == [0, 1]
    assert dataset["source_id"] == ["doc-1", "doc-2"]
    assert dataset["source_token_count"] == [4, 5]


def test_load_fineweb_requires_a_bounded_sample():
    with pytest.raises(ValueError, match="requires max_samples"):
        fineweb.load_fineweb(max_samples=None)


def test_load_fineweb_zero_samples_does_not_touch_network(monkeypatch):
    monkeypatch.setattr(
        fineweb,
        "load_dataset",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("network call")),
    )

    dataset = fineweb.load_fineweb(max_samples=0)

    assert len(dataset) == 0
