import json

from race.llm.evalscope_results import load_evalscope_generations
from race_eval.llm.evaluation.eval_corpora import _load_from_results_dir


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


def test_evalscope_generations_map_math500_level_local_indices(tmp_path):
    dataset = MiniDataset(
        [
            {"level": "Level 1"},
            {"level": "2"},
            {"level": "Level 1"},
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

    generations = load_evalscope_generations(
        str(tmp_path),
        dataset_id="math_500",
        dataset=dataset,
    )

    assert generations == {0: "A", 2: "B", 1: "C"}


def test_race_eval_results_dir_uses_math500_level_mapping(tmp_path):
    dataset = MiniDataset(
        [
            {"level": "1"},
            {"level": "2"},
            {"level": "1"},
            {"level": "2"},
        ]
    )
    (tmp_path / "math_500_Level 1.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {"index": 0, "messages": [{"role": "assistant", "content": "L1-A"}]}
                ),
                json.dumps(
                    {"index": 1, "messages": [{"role": "assistant", "content": "L1-B"}]}
                ),
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "math_500_Level 2.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {"index": 0, "messages": [{"role": "assistant", "content": "L2-A"}]}
                ),
                json.dumps(
                    {"index": 1, "messages": [{"role": "assistant", "content": "L2-B"}]}
                ),
            ]
        ),
        encoding="utf-8",
    )

    generations = _load_from_results_dir(str(tmp_path), "math_500", dataset)

    assert generations == {0: "L1-A", 2: "L1-B", 1: "L2-A", 3: "L2-B"}
