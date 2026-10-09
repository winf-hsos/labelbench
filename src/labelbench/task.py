"""Loading a task: task.yaml, labels.csv and gold.csv.

Structural problems (missing files, keys or columns) raise TaskError at once,
because nothing else can be checked without them. Content problems are
collected by labelbench.checks so that all of them are reported together.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from labelbench.checks import Issue, run_checks
from labelbench.io import read_table, read_yaml, table_columns, table_file

INVALID = "__invalid__"
"""Reserved class for predictions that are empty or not part of the label schema."""

REQUIRED_KEYS = ("name", "type", "id", "features", "label", "labels")


class TaskError(ValueError):
    """The task files are incomplete or inconsistent."""


@dataclass
class Task:
    root: Path
    spec: dict[str, Any]
    labels: list[dict[str, str]] = field(default_factory=list)
    rows: list[dict[str, str]] = field(default_factory=list)
    issues: list[Issue] = field(default_factory=list)

    @property
    def label_set(self) -> set[str]:
        return {row["label"] for row in self.labels}

    @property
    def name(self) -> str:
        return str(self.spec["name"])

    @property
    def type(self) -> str:
        return str(self.spec["type"])

    @property
    def id_col(self) -> str:
        return str(self.spec["id"])

    @property
    def label_col(self) -> str:
        return str(self.spec["label"])

    @property
    def features(self) -> list[str]:
        return list(self.spec["features"])

    @property
    def levels(self) -> list[str]:
        return list(self.spec.get("levels") or [])

    @property
    def level_gold(self) -> dict[str, str]:
        """Levels for which gold.csv carries its own per-item value: {level: column}."""
        return dict(self.spec.get("level_gold") or {})

    @property
    def show(self) -> list[str]:
        return list(self.spec.get("show") or [])

    @property
    def weight_col(self) -> str | None:
        return self.spec.get("weight")

    @property
    def split_col(self) -> str | None:
        return self.spec.get("split")

    @property
    def labels_file(self) -> Path:
        """Table spec of the label list: a CSV file or 'file.xlsx#Sheet'."""
        return self.root / self.spec["labels"]

    @property
    def gold_file(self) -> Path:
        """Table spec of the gold standard: a CSV file or 'file.xlsx#Sheet'."""
        return self.root / self.spec.get("gold", "gold.csv")

    @property
    def labels_column(self) -> str:
        """Column of the label list that holds the labels, 'label' by default."""
        return str(self.spec.get("labels_column", "label"))

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "error"]

    def level_map(self, level: str) -> dict[str, str]:
        """Map each label to its value on a coarser level, e.g. dish -> diet."""
        return {row["label"]: row[level] for row in self.labels}

    def select(self, split: str | None) -> list[dict[str, str]]:
        """Rows of one split, or all rows if no split is given or the task has none."""
        if split is None or self.split_col is None:
            return list(self.rows)
        rows = [r for r in self.rows if r[self.split_col] == split]
        if not rows:
            raise TaskError(f"No rows with {self.split_col} = {split!r} in {self.gold_file}.")
        return rows

    def items(self, rows: list[dict[str, str]]) -> list[dict[str, str]]:
        """What a classifier gets to see: the id and the feature columns, nothing else."""
        return [{"id": r[self.id_col], **{f: r[f] for f in self.features}} for r in rows]


def load_task(path: str | Path, *, strict: bool = True) -> Task:
    """Load and check a task. With strict=True, error-level issues raise TaskError."""
    root = Path(path)
    spec_file = root / "task.yaml"
    if not spec_file.is_file():
        raise TaskError(f"{spec_file} not found.")
    task = Task(root=root, spec=read_yaml(spec_file))
    _check_spec(task)
    task.labels = _read_labels(task)
    task.rows = _read_gold(task)
    task.issues = run_checks(task)

    if strict and task.errors:
        lines = [f"{task.root}: {len(task.errors)} error(s), run 'labelbench check' for details."]
        for issue in task.errors:
            lines.append(f"  [{issue.code}] {issue.message} ({issue.count}), e.g. {issue.examples[0]}")
        raise TaskError("\n".join(lines))
    return task


def _check_spec(task: Task) -> None:
    spec = task.spec
    missing = [k for k in REQUIRED_KEYS if k not in spec]
    if missing:
        raise TaskError(f"task.yaml is missing: {', '.join(missing)}.")
    if spec["type"] != "single":
        raise TaskError("Only type: single is implemented so far; type: multi is planned.")
    if not isinstance(spec["features"], list) or not spec["features"]:
        raise TaskError("features must be a non-empty list of column names.")
    stray = [lv for lv in task.level_gold if lv not in task.levels]
    if stray:
        raise TaskError(f"level_gold names levels that are not in 'levels': {stray}.")


def _read_labels(task: Task) -> list[dict[str, str]]:
    path = task.labels_file
    columns = _columns(path)
    col = task.labels_column
    if col not in columns:
        raise TaskError(f"{path} needs a column named {col!r} (set labels_column in task.yaml).")
    absent = [lv for lv in task.levels if lv not in columns]
    if absent:
        raise TaskError(f"Levels {absent} are not columns of {path}.")
    rows = _rows(path)
    if col != "label":
        # everything downstream addresses the label as row["label"]
        for r in rows:
            r["label"] = r[col]
    return rows


def _read_gold(task: Task) -> list[dict[str, str]]:
    path = task.gold_file
    needed = [task.id_col, task.label_col, *task.features, *task.show, *task.level_gold.values()]
    needed += [c for c in (task.weight_col, task.split_col) if c]
    columns = _columns(path)
    absent = [c for c in needed if c not in columns]
    if absent:
        raise TaskError(f"Columns {absent} named in task.yaml are missing in {path}.")
    return _rows(path)


def _columns(spec: Path) -> list[str]:
    if not table_file(spec).is_file():
        raise TaskError(f"{table_file(spec)} not found.")
    try:
        return table_columns(spec)
    except ValueError as exc:
        raise TaskError(str(exc)) from exc


def _rows(spec: Path) -> list[dict[str, str]]:
    try:
        return read_table(spec)
    except ValueError as exc:
        raise TaskError(str(exc)) from exc
