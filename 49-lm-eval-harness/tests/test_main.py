from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

import pytest

from main import (
    METRIC_FNS,
    SCHEMA,
    Example,
    HttpAdapter,
    TaskFormatError,
    ToyAdapter,
    UnsafeCodeError,
    code_exec,
    compile_prediction,
    exact_match,
    load_all_tasks,
    main,
    multiple_choice,
    normalize_text,
    rouge_l,
    run_leaderboard,
    run_task,
    seed_fixture_tasks,
    substring_contains,
    task_sha256,
    write_leaderboard,
)


def _example(
    identifier: str = "example-00",
    prompt: str = "prompt",
    targets: Sequence[str] = ("yes",),
    metric: str = "exact_match",
    extras: Mapping[str, object] | None = None,
) -> Example:
    return Example(
        id=identifier,
        prompt=prompt,
        targets=tuple(targets),
        metric=metric,
        extras=dict(extras or {}),
    )


class _MappingAdapter:
    def __init__(self, answers: Mapping[str, str]) -> None:
        self._answers = dict(answers)
        self.batches: list[tuple[str, ...]] = []

    @property
    def name(self) -> str:
        return "mapping"

    def generate(self, prompts: Sequence[str]) -> list[str]:
        self.batches.append(tuple(prompts))
        return [self._answers.get(prompt, "") for prompt in prompts]


class _SilentAdapter:
    @property
    def name(self) -> str:
        return "silent"

    def generate(self, prompts: Sequence[str]) -> list[str]:
        return ["" for _ in prompts]


class _ShortAdapter:
    @property
    def name(self) -> str:
        return "short"

    def generate(self, prompts: Sequence[str]) -> list[str]:
        return []


class _BareStringAdapter:
    @property
    def name(self) -> str:
        return "bare-string"

    def generate(self, prompts: Sequence[str]) -> list[str]:
        return cast(list[str], "x" * len(prompts))


def test_seed_fixture_tasks_creates_five_tasks_and_preserves_existing(tmp_path: Path) -> None:
    task_dir = tmp_path / "tasks"
    paths = seed_fixture_tasks(task_dir)
    assert len(paths) == 5
    assert {path.stem for path in paths} == {
        "arithmetic",
        "summary",
        "code-exec",
        "multiple-choice",
        "generation",
    }
    tasks = load_all_tasks(task_dir)
    assert len(tasks) == 5
    assert sum(len(examples) for examples in tasks.values()) == 16

    sentinel = "# sentinel: do not overwrite\n"
    arithmetic = task_dir / "arithmetic.jsonl"
    arithmetic.write_text(sentinel, encoding="utf-8")
    seed_fixture_tasks(task_dir)
    assert arithmetic.read_text(encoding="utf-8") == sentinel


def test_load_all_tasks_skips_comments_and_blanks(tmp_path: Path) -> None:
    task_dir = tmp_path / "tasks"
    task_dir.mkdir()
    path = task_dir / "demo.jsonl"
    path.write_text(
        "# first comment\n"
        "\n"
        '{"id": "a", "prompt": "p", "targets": ["ok"], "metric": "code_exec", '
        '"extras": {"io_pairs": [[1, 2]]}}\n'
        "   # indented comment\n"
        '{"id": "b", "prompt": "q", "targets": ["ok"], "metric": "code_exec", '
        '"extras": {"io_pairs": [[2, 3]]}}\n',
        encoding="utf-8",
    )

    tasks = load_all_tasks(task_dir)
    assert list(tasks) == ["demo"]
    first, second = tasks["demo"]
    assert first.id == "a"
    assert first.targets == ("ok",)
    assert first.extras == {"io_pairs": [[1, 2]]}
    assert second.id == "b"


@pytest.mark.parametrize(
    "content",
    [
        "not json\n",
        '{"id": "x", "prompt": "p", "targets": [], "metric": "exact_match"}\n',
        '{"id": "x", "prompt": "p", "targets": ["a"], "metric": "bogus"}\n',
        '{"id": "x", "prompt": "p", "targets": ["a"], "metric": "exact_match"}\n'
        '{"id": "y", "prompt": "q", "targets": ["b"], "metric": "substring_contains"}\n',
        '{"id": "x", "prompt": "p", "targets": ["a"], "metric": "exact_match"}\n'
        '{"id": "x", "prompt": "q", "targets": ["b"], "metric": "exact_match"}\n',
        '{"id": "x", "prompt": "p", "targets": ["ok"], "metric": "code_exec"}\n',
        '["not", "an", "object"]\n',
        "# only comments\n",
    ],
)
def test_load_all_tasks_rejects_malformed_files(tmp_path: Path, content: str) -> None:
    task_dir = tmp_path / "tasks"
    task_dir.mkdir()
    (task_dir / "bad.jsonl").write_text(content, encoding="utf-8")
    with pytest.raises(TaskFormatError, match="bad.jsonl"):
        load_all_tasks(task_dir)


def test_normalization_exact_match_and_substring() -> None:
    assert normalize_text("  Hello\n\tWORLD  ") == "hello world"
    assert exact_match("  Four \n", ["four"], {}) == 1.0
    assert exact_match("four", ["five"], {}) == 0.0
    assert substring_contains("The ROUGE-L score", ["rouge-l"], {}) == 1.0
    assert substring_contains("nothing here", [""], {}) == 0.0
    assert substring_contains("nothing here", ["missing"], {}) == 0.0


def test_multiple_choice_reads_first_letter() -> None:
    assert multiple_choice("b) Mars", ["B"], {}) == 1.0
    assert multiple_choice(" B", ["B"], {}) == 1.0
    assert multiple_choice("A", ["B"], {}) == 0.0
    assert multiple_choice("", ["B"], {}) == 0.0


def test_rouge_l_partial_overlap() -> None:
    prediction = "the cat sat on the mat"
    reference = "the cat is on the mat"
    assert rouge_l(prediction, [reference], {}) == pytest.approx(5 / 6)
    assert rouge_l(reference, [reference], {}) == 1.0
    assert rouge_l("alpha beta", ["gamma delta"], {}) == 0.0
    assert rouge_l("alpha", ["alpha", "alpha beta"], {}) == 1.0
    assert rouge_l("", [""], {}) == 1.0
    assert rouge_l("", ["x"], {}) == 0.0
    assert rouge_l("x", [], {}) == 0.0


def test_code_exec_scores_input_output_pairs() -> None:
    assert (
        code_exec(
            "def f(x):\n    return x * 2",
            ["ok"],
            {"io_pairs": [[1, 2], [3, 6], [0, 0]]},
        )
        == 1.0
    )
    assert (
        code_exec(
            "def f(x):\n    return x + 1",
            ["ok"],
            {"io_pairs": [[1, 2], [3, 9]]},
        )
        == 0.5
    )
    assert (
        code_exec(
            "def g(x):\n    return x",
            ["ok"],
            {"io_pairs": [[1, 1]]},
        )
        == 0.0
    )
    partial = "def f(x):\n    if x == 0:\n        raise ValueError('zero')\n    return x * 2"
    assert code_exec(
        partial,
        ["ok"],
        {"io_pairs": [[0, 0], [2, 4], [3, 6]]},
    ) == pytest.approx(2 / 3)


def test_code_exec_rejects_imports_and_unsafe_access() -> None:
    with pytest.raises(UnsafeCodeError):
        compile_prediction("import os\n\ndef f(x):\n    return x")
    with pytest.raises(UnsafeCodeError):
        compile_prediction("def f(x):\n    return ().__class__")
    assert (
        code_exec(
            "import os\n\ndef f(x):\n    return x",
            ["ok"],
            {"io_pairs": [[1, 1]]},
        )
        == 0.0
    )
    assert (
        code_exec(
            "def f(x):\n    return open('/etc/passwd').read()",
            ["ok"],
            {"io_pairs": [[1, 1]]},
        )
        == 0.0
    )


def test_code_exec_compares_bool_strictly() -> None:
    assert (
        code_exec(
            "def f(x):\n    return True",
            ["ok"],
            {"io_pairs": [[1, 1]]},
        )
        == 0.0
    )
    assert (
        code_exec(
            "def f(x):\n    return [x, {'a': x}]",
            ["ok"],
            {"io_pairs": [[1, [1, {"a": 1}]]]},
        )
        == 1.0
    )


def test_toy_adapter_scores_every_fixture_perfectly(tmp_path: Path) -> None:
    task_dir = tmp_path / "tasks"
    seed_fixture_tasks(task_dir)
    board = run_leaderboard(load_all_tasks(task_dir), ToyAdapter(), batch_size=3)
    assert board.schema == SCHEMA
    assert board.overall == 1.0
    assert set(board.tasks) == {
        "arithmetic",
        "summary",
        "code-exec",
        "multiple-choice",
        "generation",
    }
    assert sum(result.total for result in board.tasks.values()) == 16
    assert all(result.score == 1.0 for result in board.tasks.values())
    assert all(result.correct == result.total for result in board.tasks.values())


def test_silent_adapter_scores_zero(tmp_path: Path) -> None:
    task_dir = tmp_path / "tasks"
    seed_fixture_tasks(task_dir)
    board = run_leaderboard(load_all_tasks(task_dir), _SilentAdapter(), batch_size=2)
    assert board.overall == 0.0
    assert all(result.score == 0.0 for result in board.tasks.values())


def test_runner_batches_in_fixed_sizes() -> None:
    examples = [_example(f"e{index}", f"p{index}") for index in range(7)]
    adapter = _MappingAdapter({f"p{index}": "yes" for index in range(7)})
    result = run_task("demo", examples, adapter, batch_size=3)
    assert [len(batch) for batch in adapter.batches] == [3, 3, 1]
    assert result.task == "demo"
    assert result.score == 1.0
    assert result.correct == 7
    assert result.total == 7


def test_runner_rejects_bad_batch_size_and_short_adapter_output() -> None:
    examples = [_example()]
    with pytest.raises(ValueError, match="batch_size"):
        run_task("demo", examples, _MappingAdapter({"prompt": "yes"}), batch_size=0)
    with pytest.raises(ValueError, match="returned 0 predictions"):
        run_task("demo", examples, _ShortAdapter(), batch_size=1)
    with pytest.raises(ValueError, match="bare string"):
        run_task("demo", examples, _BareStringAdapter(), batch_size=1)


def test_overall_is_macro_average_of_task_scores() -> None:
    tasks = {
        "a": [_example(f"a{index}", f"a{index}") for index in range(4)],
        "b": [_example("b0", "b0")],
    }
    answers = {f"a{index}": "yes" for index in range(4)}
    answers["b0"] = "no"
    board = run_leaderboard(tasks, _MappingAdapter(answers), batch_size=2)
    assert board.tasks["a"].score == 1.0
    assert board.tasks["b"].score == 0.0
    assert board.overall == 0.5


def test_task_hash_tracks_semantic_content(tmp_path: Path) -> None:
    first = [_example("a", "p", ("x",))]
    second = [_example("a", "p", ("y",))]
    assert task_sha256(first) == task_sha256(first)
    assert task_sha256(first) != task_sha256(second)

    task_dir = tmp_path / "tasks"
    task_dir.mkdir()
    path = task_dir / "demo.jsonl"
    record = '{"id": "a", "prompt": "p", "targets": ["x"], "metric": "exact_match"}\n'
    path.write_text(f"# first\n\n{record}", encoding="utf-8")
    hash_one = task_sha256(load_all_tasks(task_dir)["demo"])
    path.write_text(f"# second\n\n{record}", encoding="utf-8")
    hash_two = task_sha256(load_all_tasks(task_dir)["demo"])
    assert hash_one == hash_two


def test_write_leaderboard_schema_and_per_example_flag(tmp_path: Path) -> None:
    tasks = {"demo": [_example("e0", "p0")]}
    board = run_leaderboard(tasks, _MappingAdapter({"p0": "yes"}))
    plain = tmp_path / "plain.json"
    write_leaderboard(board, plain)
    payload = json.loads(plain.read_text(encoding="utf-8"))
    assert payload["schema"] == SCHEMA
    assert payload["adapter"] == "mapping"
    assert payload["task_count"] == 1
    assert "examples" not in payload["tasks"]["demo"]
    assert len(payload["tasks"]["demo"]["task_sha256"]) == 64

    first_bytes = plain.read_bytes()
    write_leaderboard(board, plain)
    assert plain.read_bytes() == first_bytes

    detailed = tmp_path / "detailed.json"
    write_leaderboard(board, detailed, include_per_example=True)
    detailed_payload = json.loads(detailed.read_text(encoding="utf-8"))
    example = detailed_payload["tasks"]["demo"]["examples"][0]
    assert example["id"] == "e0"
    assert example["prediction"] == "yes"
    assert example["targets"] == ["yes"]
    assert example["score"] == 1.0


def test_main_seeds_runs_and_writes_skill_recipe(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    task_dir = tmp_path / "tasks"
    output = tmp_path / "out" / "leaderboard.json"
    args = [
        "--task-dir",
        str(task_dir),
        "--output",
        str(output),
        "--batch-size",
        "3",
        "--include-per-example",
    ]
    assert main(args) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema"] == SCHEMA
    assert payload["overall"] == 1.0
    assert payload["task_count"] == 5
    assert all("examples" in task for task in payload["tasks"].values())

    skill = output.parent / "skill-lm-eval-harness.md"
    assert skill.is_file()
    skill_text = skill.read_text(encoding="utf-8")
    assert SCHEMA in skill_text
    for name in METRIC_FNS:
        assert name in skill_text

    captured = capsys.readouterr()
    assert "overall: 1.000" in captured.out
    assert main(args) == 0


def test_http_adapter_posts_and_parses_text() -> None:
    calls: list[tuple[str, dict[str, str], bytes, float]] = []

    def transport(
        url: str,
        headers: Mapping[str, str],
        body: bytes,
        timeout: float,
    ) -> bytes:
        calls.append((url, dict(headers), body, timeout))
        return b'{"text": "answer"}'

    adapter = HttpAdapter(
        "https://example.test/generate",
        api_key="test_only",
        model="demo",
        timeout=2.5,
        transport=transport,
    )
    assert adapter.name == "http:demo"
    assert adapter.generate(["first", "second"]) == ["answer", "answer"]
    assert len(calls) == 2
    url, headers, body, timeout = calls[0]
    assert url == "https://example.test/generate"
    assert headers["Authorization"] == "Bearer test_only"
    assert headers["Content-Type"] == "application/json"
    assert json.loads(body) == {"model": "demo", "prompt": "first"}
    assert timeout == 2.5


def test_http_adapter_rejects_malformed_response() -> None:
    def missing_text(
        url: str,
        headers: Mapping[str, str],
        body: bytes,
        timeout: float,
    ) -> bytes:
        return b'{"answer": "missing text"}'

    adapter = HttpAdapter("https://example.test/generate", transport=missing_text)
    with pytest.raises(ValueError, match="string 'text'"):
        adapter.generate(["prompt"])

    def invalid_json(
        url: str,
        headers: Mapping[str, str],
        body: bytes,
        timeout: float,
    ) -> bytes:
        return b"not json"

    bad_adapter = HttpAdapter(
        "https://example.test/generate",
        transport=invalid_json,
    )
    with pytest.raises(ValueError):
        bad_adapter.generate(["prompt"])


def test_load_all_tasks_missing_directory_and_empty_leaderboard(tmp_path: Path) -> None:
    with pytest.raises(TaskFormatError, match="not found"):
        load_all_tasks(tmp_path / "missing")
    with pytest.raises(ValueError, match="no tasks"):
        run_leaderboard({}, ToyAdapter())
