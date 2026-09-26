from __future__ import annotations

import argparse
import ast
import builtins
import hashlib
import json
import operator
import re
import sys
import time
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import CodeType
from typing import Protocol, cast

SCHEMA = "lm-eval-harness/v1"
SKILL_FILENAME = "skill-lm-eval-harness.md"
DEFAULT_BATCH_SIZE = 8
MODULE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TASK_DIR = MODULE_ROOT / "outputs" / "tasks"
DEFAULT_LEADERBOARD_PATH = MODULE_ROOT / "outputs" / "leaderboard.json"

MetricFn = Callable[[str, Sequence[str], Mapping[str, object]], float]


class TaskFormatError(ValueError):
    """A JSONL task file does not satisfy the task contract."""


class UnsafeCodeError(ValueError):
    """A code prediction uses syntax that the restricted evaluator rejects."""


@dataclass(frozen=True)
class Example:
    id: str
    prompt: str
    targets: tuple[str, ...]
    metric: str
    extras: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ExampleResult:
    id: str
    prompt: str
    targets: tuple[str, ...]
    prediction: str
    score: float

    def to_json_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "prediction": self.prediction,
            "prompt": self.prompt,
            "score": self.score,
            "targets": list(self.targets),
        }


@dataclass(frozen=True)
class TaskResult:
    task: str
    metric: str
    score: float
    correct: int
    total: int
    latency_ms: float
    task_sha256: str
    examples: tuple[ExampleResult, ...]

    def to_json_dict(self, include_per_example: bool) -> dict[str, object]:
        payload: dict[str, object] = {
            "correct": self.correct,
            "latency_ms": self.latency_ms,
            "metric": self.metric,
            "score": self.score,
            "task_sha256": self.task_sha256,
            "total": self.total,
        }
        if include_per_example:
            payload["examples"] = [example.to_json_dict() for example in self.examples]
        return payload


@dataclass(frozen=True)
class Leaderboard:
    schema: str
    adapter: str
    batch_size: int
    overall: float
    tasks: dict[str, TaskResult]

    def to_json_dict(self, include_per_example: bool) -> dict[str, object]:
        tasks = {
            name: result.to_json_dict(include_per_example)
            for name, result in self.tasks.items()
        }
        return {
            "adapter": self.adapter,
            "batch_size": self.batch_size,
            "overall": self.overall,
            "schema": self.schema,
            "task_count": len(self.tasks),
            "tasks": tasks,
        }


FIXTURE_TASKS: dict[str, list[dict[str, object]]] = {
    "arithmetic": [
        {"id": "arith-00", "prompt": "compute: 2 + 2", "targets": ["4"], "metric": "exact_match"},
        {"id": "arith-01", "prompt": "compute: 17 - 9", "targets": ["8"], "metric": "exact_match"},
        {"id": "arith-02", "prompt": "compute: 6 * 7", "targets": ["42"], "metric": "exact_match"},
        {"id": "arith-03", "prompt": "compute: 144 / 12", "targets": ["12"], "metric": "exact_match"},
    ],
    "summary": [
        {
            "id": "summary-00",
            "prompt": "summarize: Rain fell all morning. By noon the streets were flooded and traffic stopped.",
            "targets": ["morning rain flooded the streets and stopped traffic"],
            "metric": "rouge_l",
        },
        {
            "id": "summary-01",
            "prompt": "summarize: The library extended its hours. Students can now study until midnight.",
            "targets": ["the library extended its hours so students can study until midnight"],
            "metric": "rouge_l",
        },
        {
            "id": "summary-02",
            "prompt": "summarize: The team fixed two bugs, reran the tests, and shipped the release.",
            "targets": ["the team fixed two bugs and shipped the release after tests passed"],
            "metric": "rouge_l",
        },
    ],
    "code-exec": [
        {
            "id": "code-00",
            "prompt": "python: write a function f that doubles its input",
            "targets": ["ok"],
            "metric": "code_exec",
            "extras": {"io_pairs": [[1, 2], [3, 6], [0, 0]]},
        },
        {
            "id": "code-01",
            "prompt": "python: write a function f that squares its input",
            "targets": ["ok"],
            "metric": "code_exec",
            "extras": {"io_pairs": [[2, 4], [5, 25], [-3, 9]]},
        },
        {
            "id": "code-02",
            "prompt": "python: write a function f that adds one to its input",
            "targets": ["ok"],
            "metric": "code_exec",
            "extras": {"io_pairs": [[0, 1], [-1, 0], [41, 42]]},
        },
    ],
    "multiple-choice": [
        {
            "id": "mc-00",
            "prompt": "Which planet is known as the Red Planet?\nA) Venus\nB) Mars\nC) Jupiter\nD) Saturn\nAnswer with the letter only.",
            "targets": ["B"],
            "metric": "multiple_choice",
        },
        {
            "id": "mc-01",
            "prompt": "What is the capital of France?\nA) Rome\nB) Berlin\nC) Paris\nD) Madrid\nAnswer with the letter only.",
            "targets": ["C"],
            "metric": "multiple_choice",
        },
        {
            "id": "mc-02",
            "prompt": "Which data structure uses first-in, first-out order?\nA) stack\nB) queue\nC) tree\nD) graph\nAnswer with the letter only.",
            "targets": ["B"],
            "metric": "multiple_choice",
        },
    ],
    "generation": [
        {
            "id": "gen-00",
            "prompt": "generate: a one-line slogan for a tool that runs model evaluations offline",
            "targets": ["offline"],
            "metric": "substring_contains",
        },
        {
            "id": "gen-01",
            "prompt": "generate: name the metric that compares a prediction to a reference by longest common subsequence",
            "targets": ["rouge-l"],
            "metric": "substring_contains",
        },
        {
            "id": "gen-02",
            "prompt": "generate: explain in one line why task files must be pinned",
            "targets": ["same tasks"],
            "metric": "substring_contains",
        },
    ],
}

_TOY_ANSWERS: dict[str, str] = {
    "summarize: Rain fell all morning. By noon the streets were flooded and traffic stopped.":
        "morning rain flooded the streets and stopped traffic",
    "summarize: The library extended its hours. Students can now study until midnight.":
        "the library extended its hours so students can study until midnight",
    "summarize: The team fixed two bugs, reran the tests, and shipped the release.":
        "the team fixed two bugs and shipped the release after tests passed",
    "python: write a function f that doubles its input": "def f(x):\n    return x * 2",
    "python: write a function f that squares its input": "def f(x):\n    return x * x",
    "python: write a function f that adds one to its input": "def f(x):\n    return x + 1",
    "Which planet is known as the Red Planet?\nA) Venus\nB) Mars\nC) Jupiter\nD) Saturn\nAnswer with the letter only.": "B",
    "What is the capital of France?\nA) Rome\nB) Berlin\nC) Paris\nD) Madrid\nAnswer with the letter only.": "C",
    "Which data structure uses first-in, first-out order?\nA) stack\nB) queue\nC) tree\nD) graph\nAnswer with the letter only.": "B",
    "generate: a one-line slogan for a tool that runs model evaluations offline":
        "Run every model evaluation offline, reproducibly.",
    "generate: name the metric that compares a prediction to a reference by longest common subsequence":
        "The reference comparison uses ROUGE-L.",
    "generate: explain in one line why task files must be pinned":
        "Pin the same tasks so scores stay comparable.",
}

_ARITHMETIC_RE = re.compile(r"^compute:\s*(-?\d+)\s*([+\-*/])\s*(-?\d+)$")
_ARITHMETIC_OPS: dict[str, Callable[[int, int], int | float]] = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
    "/": operator.truediv,
}


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def seed_fixture_tasks(target_dir: Path) -> list[Path]:
    target_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for task_name, records in FIXTURE_TASKS.items():
        path = target_dir / f"{task_name}.jsonl"
        written.append(path)
        if path.exists():
            continue
        lines = [
            f"# {task_name}: fixture task; lines beginning with # are comments.",
            "",
            *[json.dumps(record, ensure_ascii=False) for record in records],
        ]
        _atomic_write_text(path, "\n".join(lines) + "\n")
    return written


def _require_non_empty_string(value: object, field_name: str, path: Path, line_number: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TaskFormatError(
            f"{path.name}:{line_number}: {field_name} must be a non-empty string"
        )
    return value


def _parse_io_pairs(value: object) -> list[tuple[object, object]]:
    if not isinstance(value, list) or not value:
        raise ValueError("io_pairs must be a non-empty list")
    pairs: list[tuple[object, object]] = []
    for index, item in enumerate(value):
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError(f"io_pairs[{index}] must contain exactly two values")
        pairs.append((item[0], item[1]))
    return pairs


def _parse_example(record: Mapping[str, object], path: Path, line_number: int) -> Example:
    identifier = _require_non_empty_string(record.get("id"), "id", path, line_number)
    prompt = _require_non_empty_string(record.get("prompt"), "prompt", path, line_number)
    raw_targets = record.get("targets")
    if not isinstance(raw_targets, list) or not raw_targets:
        raise TaskFormatError(
            f"{path.name}:{line_number}: targets must be a non-empty list"
        )
    targets = tuple(
        _require_non_empty_string(target, f"targets[{index}]", path, line_number)
        for index, target in enumerate(raw_targets)
    )
    metric = _require_non_empty_string(record.get("metric"), "metric", path, line_number)
    if metric not in METRIC_FNS:
        raise TaskFormatError(
            f"{path.name}:{line_number}: unknown metric {metric!r}"
        )
    raw_extras = record.get("extras", {})
    if not isinstance(raw_extras, dict):
        raise TaskFormatError(
            f"{path.name}:{line_number}: extras must be a JSON object"
        )
    extras: dict[str, object] = {}
    for key, value in raw_extras.items():
        if not isinstance(key, str):
            raise TaskFormatError(
                f"{path.name}:{line_number}: extras keys must be strings"
            )
        extras[key] = value
    if metric == "code_exec":
        try:
            _parse_io_pairs(extras.get("io_pairs"))
        except ValueError as exc:
            raise TaskFormatError(
                f"{path.name}:{line_number}: invalid code_exec extras: {exc}"
            ) from exc
    return Example(
        id=identifier,
        prompt=prompt,
        targets=targets,
        metric=metric,
        extras=extras,
    )


def load_all_tasks(task_dir: Path) -> dict[str, list[Example]]:
    if not task_dir.is_dir():
        raise TaskFormatError(f"task directory not found: {task_dir}")
    tasks: dict[str, list[Example]] = {}
    for path in sorted(task_dir.glob("*.jsonl")):
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise TaskFormatError(f"{path.name}: task file is not UTF-8") from exc
        except OSError as exc:
            raise TaskFormatError(f"{path.name}: cannot read task file: {exc}") from exc
        examples: list[Example] = []
        seen_ids: set[str] = set()
        metric_name: str | None = None
        for line_number, raw_line in enumerate(text.splitlines(), start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                record: object = json.loads(line)
            except json.JSONDecodeError as exc:
                raise TaskFormatError(
                    f"{path.name}:{line_number}: invalid JSON: {exc.msg}"
                ) from exc
            if not isinstance(record, dict):
                raise TaskFormatError(
                    f"{path.name}:{line_number}: example must be a JSON object"
                )
            example = _parse_example(record, path, line_number)
            if metric_name is None:
                metric_name = example.metric
            elif metric_name != example.metric:
                raise TaskFormatError(
                    f"{path.name}:{line_number}: mixed metrics "
                    f"{metric_name!r} and {example.metric!r}"
                )
            if example.id in seen_ids:
                raise TaskFormatError(
                    f"{path.name}:{line_number}: duplicate example id {example.id!r}"
                )
            seen_ids.add(example.id)
            examples.append(example)
        if not examples:
            raise TaskFormatError(f"{path.name}: task file has no examples")
        tasks[path.stem] = examples
    return tasks


def normalize_text(text: str) -> str:
    return " ".join(text.lower().split())


def exact_match(
    prediction: str,
    targets: Sequence[str],
    extras: Mapping[str, object],
) -> float:
    del extras
    normalized = normalize_text(prediction)
    return 1.0 if any(normalized == normalize_text(target) for target in targets) else 0.0


def substring_contains(
    prediction: str,
    targets: Sequence[str],
    extras: Mapping[str, object],
) -> float:
    del extras
    normalized = normalize_text(prediction)
    for target in targets:
        candidate = normalize_text(target)
        if candidate and candidate in normalized:
            return 1.0
    return 0.0


def multiple_choice(
    prediction: str,
    targets: Sequence[str],
    extras: Mapping[str, object],
) -> float:
    del extras
    answer = prediction.strip()[:1].upper()
    if not answer:
        return 0.0
    allowed = {
        target.strip()[:1].upper()
        for target in targets
        if target.strip()
    }
    return 1.0 if answer in allowed else 0.0


def _lcs_length(left: Sequence[str], right: Sequence[str]) -> int:
    if not left or not right:
        return 0
    previous = [0] * (len(right) + 1)
    for left_token in left:
        current = [0] * (len(right) + 1)
        for index, right_token in enumerate(right, start=1):
            if left_token == right_token:
                current[index] = previous[index - 1] + 1
            else:
                current[index] = max(previous[index], current[index - 1])
        previous = current
    return previous[-1]


def rouge_l(
    prediction: str,
    targets: Sequence[str],
    extras: Mapping[str, object],
) -> float:
    del extras
    prediction_tokens = normalize_text(prediction).split()
    best = 0.0
    for target in targets:
        reference_tokens = normalize_text(target).split()
        if not prediction_tokens and not reference_tokens:
            score = 1.0
        elif not prediction_tokens or not reference_tokens:
            score = 0.0
        else:
            lcs = _lcs_length(prediction_tokens, reference_tokens)
            if lcs == 0:
                score = 0.0
            else:
                precision = lcs / len(prediction_tokens)
                recall = lcs / len(reference_tokens)
                score = 2.0 * precision * recall / (precision + recall)
        if score > best:
            best = score
    return best


_SAFE_BUILTIN_NAMES = (
    "abs",
    "all",
    "any",
    "bin",
    "bool",
    "chr",
    "dict",
    "divmod",
    "enumerate",
    "filter",
    "float",
    "format",
    "frozenset",
    "hex",
    "int",
    "isinstance",
    "issubclass",
    "iter",
    "len",
    "list",
    "map",
    "max",
    "min",
    "next",
    "oct",
    "ord",
    "pow",
    "range",
    "repr",
    "reversed",
    "round",
    "set",
    "slice",
    "sorted",
    "str",
    "sum",
    "tuple",
    "zip",
    "Exception",
    "ValueError",
    "TypeError",
    "ZeroDivisionError",
    "IndexError",
    "KeyError",
)
SAFE_BUILTINS: dict[str, object] = {
    name: getattr(builtins, name) for name in _SAFE_BUILTIN_NAMES
}


def compile_prediction(source: str) -> CodeType:
    tree = ast.parse(source, filename="<prediction>", mode="exec")
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise UnsafeCodeError("imports are not allowed in code predictions")
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            raise UnsafeCodeError(f"dunder name {node.id!r} is not allowed")
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise UnsafeCodeError(f"dunder attribute {node.attr!r} is not allowed")
    return compile(tree, "<prediction>", "exec")


def _exec_prediction(source: str) -> Mapping[str, object]:
    namespace: dict[str, object] = {
        "__builtins__": dict(SAFE_BUILTINS),
        "__name__": "prediction",
    }
    exec(compile_prediction(source), namespace)
    return namespace


def _values_equal(actual: object, expected: object) -> bool:
    if isinstance(actual, bool) or isinstance(expected, bool):
        return (
            isinstance(actual, bool)
            and isinstance(expected, bool)
            and actual == expected
        )
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        return actual == expected
    if isinstance(actual, list) and isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _values_equal(left, right)
            for left, right in zip(actual, expected, strict=True)
        )
    if isinstance(actual, dict) and isinstance(expected, dict):
        if actual.keys() != expected.keys():
            return False
        return all(
            _values_equal(actual[key], expected[key]) for key in actual
        )
    return type(actual) is type(expected) and actual == expected


def code_exec(
    prediction: str,
    targets: Sequence[str],
    extras: Mapping[str, object],
) -> float:
    del targets
    try:
        pairs = _parse_io_pairs(extras.get("io_pairs"))
        namespace = _exec_prediction(prediction)
        candidate = namespace.get("f")
        if not callable(candidate):
            return 0.0
        function = cast(Callable[[object], object], candidate)
        matches = 0
        for input_value, expected in pairs:
            try:
                actual = function(input_value)
            except Exception:
                continue
            if _values_equal(actual, expected):
                matches += 1
        return matches / len(pairs)
    except Exception:
        return 0.0


METRIC_FNS: dict[str, MetricFn] = {
    "exact_match": exact_match,
    "substring_contains": substring_contains,
    "multiple_choice": multiple_choice,
    "rouge_l": rouge_l,
    "code_exec": code_exec,
}


class ModelAdapter(Protocol):
    @property
    def name(self) -> str: ...

    def generate(self, prompts: Sequence[str]) -> list[str]: ...


def _toy_arithmetic(prompt: str) -> str:
    match = _ARITHMETIC_RE.fullmatch(prompt)
    if match is None:
        return ""
    left = int(match.group(1))
    operation = match.group(2)
    right = int(match.group(3))
    try:
        value = float(_ARITHMETIC_OPS[operation](left, right))
    except ZeroDivisionError:
        return ""
    if value.is_integer():
        return str(int(value))
    return f"{value:g}"


class ToyAdapter:
    @property
    def name(self) -> str:
        return "toy"

    def generate(self, prompts: Sequence[str]) -> list[str]:
        return [self._answer(prompt) for prompt in prompts]

    def _answer(self, prompt: str) -> str:
        if prompt.startswith("compute:"):
            return _toy_arithmetic(prompt)
        return _TOY_ANSWERS.get(prompt, "")


HttpTransport = Callable[[str, Mapping[str, str], bytes, float], bytes]


def _urllib_transport(
    url: str,
    headers: Mapping[str, str],
    body: bytes,
    timeout: float,
) -> bytes:
    request = urllib.request.Request(
        url,
        data=body,
        headers=dict(headers),
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return cast(bytes, response.read())


class HttpAdapter:
    def __init__(
        self,
        endpoint: str,
        api_key: str | None = None,
        *,
        model: str = "default",
        timeout: float = 30.0,
        transport: HttpTransport | None = None,
    ) -> None:
        if not endpoint:
            raise ValueError("endpoint must not be empty")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self._endpoint = endpoint
        self._api_key = api_key
        self._model = model
        self._timeout = timeout
        self._transport = transport if transport is not None else _urllib_transport

    @property
    def name(self) -> str:
        return f"http:{self._model}"

    def generate(self, prompts: Sequence[str]) -> list[str]:
        answers: list[str] = []
        for prompt in prompts:
            body = json.dumps({"model": self._model, "prompt": prompt}).encode("utf-8")
            headers = {
                "Accept": "application/json",
                "Content-Type": "application/json",
            }
            if self._api_key is not None:
                headers["Authorization"] = f"Bearer {self._api_key}"
            raw = self._transport(self._endpoint, headers, body, self._timeout)
            parsed: object = json.loads(raw)
            if not isinstance(parsed, dict):
                raise ValueError("http adapter expected a JSON object response")
            text: object = parsed.get("text")
            if not isinstance(text, str):
                raise ValueError("http adapter expected a string 'text' field")
            answers.append(text)
        return answers


def task_sha256(examples: Sequence[Example]) -> str:
    payload = [
        {
            "extras": dict(example.extras),
            "id": example.id,
            "metric": example.metric,
            "prompt": example.prompt,
            "targets": list(example.targets),
        }
        for example in examples
    ]
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def run_task(
    name: str,
    examples: Sequence[Example],
    adapter: ModelAdapter,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> TaskResult:
    if not examples:
        raise ValueError(f"task {name!r} has no examples")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    metric_names = {example.metric for example in examples}
    if len(metric_names) != 1:
        raise ValueError(f"task {name!r} mixes metrics: {sorted(metric_names)!r}")
    metric_name = next(iter(metric_names))
    try:
        metric = METRIC_FNS[metric_name]
    except KeyError as exc:
        raise ValueError(f"task {name!r} uses unknown metric {metric_name!r}") from exc

    started = time.perf_counter()
    results: list[ExampleResult] = []
    for offset in range(0, len(examples), batch_size):
        batch = list(examples[offset:offset + batch_size])
        prompts = [example.prompt for example in batch]
        raw_predictions = adapter.generate(prompts)
        if isinstance(raw_predictions, str):
            raise ValueError(
                f"adapter {adapter.name!r} returned a bare string for task {name!r}"
            )
        predictions = list(raw_predictions)
        if len(predictions) != len(batch):
            raise ValueError(
                f"adapter {adapter.name!r} returned {len(predictions)} predictions "
                f"for {len(batch)} prompts in task {name!r}"
            )
        for example, prediction in zip(batch, predictions, strict=True):
            if not isinstance(prediction, str):
                raise ValueError(
                    f"adapter {adapter.name!r} returned a non-string prediction "
                    f"in task {name!r}"
                )
            score = float(metric(prediction, example.targets, example.extras))
            if not 0.0 <= score <= 1.0:
                raise ValueError(
                    f"metric {metric_name!r} returned {score!r}; expected [0, 1]"
                )
            results.append(
                ExampleResult(
                    id=example.id,
                    prompt=example.prompt,
                    targets=example.targets,
                    prediction=prediction,
                    score=score,
                )
            )
    latency_ms = (time.perf_counter() - started) * 1000.0
    score = sum(result.score for result in results) / len(results)
    correct = sum(1 for result in results if result.score == 1.0)
    return TaskResult(
        task=name,
        metric=metric_name,
        score=score,
        correct=correct,
        total=len(results),
        latency_ms=latency_ms,
        task_sha256=task_sha256(examples),
        examples=tuple(results),
    )


def run_leaderboard(
    tasks: Mapping[str, Sequence[Example]],
    adapter: ModelAdapter,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> Leaderboard:
    if not tasks:
        raise ValueError("no tasks to run")
    results: dict[str, TaskResult] = {}
    for name in sorted(tasks):
        results[name] = run_task(name, tasks[name], adapter, batch_size=batch_size)
    overall = sum(result.score for result in results.values()) / len(results)
    return Leaderboard(
        schema=SCHEMA,
        adapter=adapter.name,
        batch_size=batch_size,
        overall=overall,
        tasks=results,
    )


def write_leaderboard(
    leaderboard: Leaderboard,
    path: Path,
    *,
    include_per_example: bool = False,
) -> Path:
    payload = leaderboard.to_json_dict(include_per_example)
    _atomic_write_text(
        path,
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
    )
    return path


def write_skill_recipe(path: Path) -> Path:
    metric_lines = "\n".join(
        f"- `{name}`" for name in METRIC_FNS
    )
    text = f"""# Skill: language model evaluation harness

Use this recipe to add a small, auditable evaluation harness to a project.

## Task contract

Each task is a `.jsonl` file whose name is the task name. Blank lines and lines beginning with
`#` are comments. Every other line is one example:

```json
{{"id": "arith-00", "prompt": "compute: 2 + 2", "targets": ["4"], "metric": "exact_match"}}
```

For executable scoring, `extras` carries the input-output pairs:

```json
{{"id": "code-00", "prompt": "python: write a function f that doubles its input", "targets": ["ok"], "metric": "code_exec", "extras": {{"io_pairs": [[1, 2], [3, 6]]}}}}
```

## Metrics

Each metric is a function from `(prediction, targets, extras)` to a float in `[0, 1]`:

{metric_lines}

`rouge_l` is token-level LCS F1 (beta = 1). `code_exec` executes the prediction with an allowlisted
builtins namespace and rejects imports and dunder access; it calls `f(input)` for every pair.

## Adapter seam

The only model-specific code is an object with:

```python
class ModelAdapter(Protocol):
    @property
    def name(self) -> str: ...
    def generate(self, prompts: Sequence[str]) -> list[str]: ...
```

`ToyAdapter` is a deterministic fixture matcher. `HttpAdapter` posts
`{{"model": ..., "prompt": ...}}` and expects `{{"text": ...}}`.

## Runner and leaderboard

`run_task` calls `generate` once per batch of at most `batch_size` prompts, scores each example,
and reports the mean, full-credit count, total, and latency. `run_leaderboard` takes the macro
average of task scores. The output schema is `{SCHEMA}`. Each task carries a canonical
`task_sha256` over its parsed content so a score can be tied to the task that produced it.
Pass `--include-per-example` to retain predictions for diffing.

## Safety

`code_exec` is a restricted-namespace metric, not a security sandbox. Run model-generated code in a
separate process or container with a timeout before using it on untrusted output.
"""
    _atomic_write_text(path, text)
    return path


def _positive_int(raw: str) -> int:
    value = int(raw)
    if value < 1:
        raise argparse.ArgumentTypeError("batch size must be at least 1")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the JSONL evaluation harness with the deterministic toy adapter.",
    )
    parser.add_argument(
        "--task-dir",
        type=Path,
        default=DEFAULT_TASK_DIR,
        help="directory containing *.jsonl task files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_LEADERBOARD_PATH,
        help="leaderboard JSON path",
    )
    parser.add_argument(
        "--batch-size",
        type=_positive_int,
        default=DEFAULT_BATCH_SIZE,
        help="maximum prompts per adapter call",
    )
    parser.add_argument(
        "--include-per-example",
        action="store_true",
        help="include predictions and per-example scores in the leaderboard",
    )
    args = parser.parse_args(argv)

    try:
        if not any(args.task_dir.glob("*.jsonl")):
            seed_fixture_tasks(args.task_dir)
            print(f"seeded {len(FIXTURE_TASKS)} tasks in {args.task_dir}")
        tasks = load_all_tasks(args.task_dir)
        leaderboard = run_leaderboard(
            tasks,
            ToyAdapter(),
            batch_size=args.batch_size,
        )
        output_path = write_leaderboard(
            leaderboard,
            args.output,
            include_per_example=args.include_per_example,
        )
        skill_path = write_skill_recipe(args.output.parent / SKILL_FILENAME)
    except (TaskFormatError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(
        f"adapter: {leaderboard.adapter}  "
        f"batch_size: {leaderboard.batch_size}  "
        f"tasks: {len(leaderboard.tasks)}"
    )
    print(f"{'task':<18} {'metric':<20} {'score':>6} {'correct':>9} {'latency_ms':>10}")
    for name, result in leaderboard.tasks.items():
        print(
            f"{name:<18} {result.metric:<20} {result.score:6.3f} "
            f"{result.correct:>4}/{result.total:<4} {result.latency_ms:10.3f}"
        )
    print(f"overall: {leaderboard.overall:.3f}")
    print(f"wrote: {output_path}")
    print(f"wrote: {skill_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
