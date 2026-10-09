"""Consistency checks for a task's label schema and gold standard.

Every check returns issues instead of raising, so `labelbench check` can show
all problems at once. Severity decides what happens next:

- error: the task cannot be evaluated reliably; `run` refuses to start.
- warning: evaluation works, but results may be distorted; shown in every report.
- info: worth knowing, no action required.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher, get_close_matches
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from labelbench.task import Task

SEVERITIES = ("error", "warning", "info")
MAX_EXAMPLES = 25
SIMILAR_RATIO = 0.9
SIMILAR_MAX_LABELS = 3000
KNOWN_SPLITS = {"dev", "test"}


@dataclass
class Issue:
    severity: str
    code: str
    message: str
    count: int = 0
    examples: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_checks(task: Task) -> list[Issue]:
    from labelbench.task import INVALID

    issues: list[Issue] = []
    issues += _check_labels(task, INVALID)
    issues += _check_gold(task)
    issues += _check_levels(task)
    issues += _check_duplicates(task)
    order = {s: i for i, s in enumerate(SEVERITIES)}
    return sorted(issues, key=lambda i: order[i.severity])


def _issue(severity: str, code: str, message: str, examples: list[str]) -> list[Issue]:
    if not examples:
        return []
    return [Issue(severity, code, message, len(examples), examples[:MAX_EXAMPLES])]


# --- label schema -----------------------------------------------------------

def _check_labels(task: Task, invalid: str) -> list[Issue]:
    names = [r["label"] for r in task.labels]
    counts = Counter(names)
    issues = _issue("error", "duplicate_label",
                    "Labels that appear more than once in labels.csv.",
                    [f"{n!r} ({c}x)" for n, c in counts.items() if c > 1])
    issues += _issue("error", "empty_label", "Rows in labels.csv without a label.",
                     [f"row {i + 2}" for i, n in enumerate(names) if not n.strip()])
    issues += _issue("error", "reserved_label",
                     f"{invalid} is reserved for invalid predictions.",
                     [n for n in names if n == invalid])
    issues += _issue("warning", "label_whitespace",
                     "Labels with leading or trailing whitespace; they only match "
                     "gold labels with exactly the same whitespace.",
                     [repr(n) for n in names if n != n.strip()])
    issues += _similar_labels(names)
    used = {r[task.label_col] for r in task.rows}
    issues += _issue("info", "unused_labels",
                     "Labels that never occur in the gold standard, so their quality "
                     "cannot be measured.",
                     [n for n in names if n not in used])
    return issues


def _normalise(label: str) -> str:
    return re.sub(r"[\W_]+", "", label.casefold())


def _similar_labels(names: list[str]) -> list[Issue]:
    unique = sorted(set(names))
    if len(unique) > SIMILAR_MAX_LABELS:
        return [Issue("info", "similar_labels_skipped",
                      f"Similarity check skipped for more than {SIMILAR_MAX_LABELS} labels.")]
    pairs = []
    for i, a in enumerate(unique):
        na = _normalise(a)
        for b in unique[i + 1:]:
            nb = _normalise(b)
            if na == nb:
                pairs.append(f"{a!r} ~ {b!r} (identical apart from case/punctuation)")
                continue
            m = SequenceMatcher(None, na, nb)
            if m.real_quick_ratio() >= SIMILAR_RATIO and m.quick_ratio() >= SIMILAR_RATIO \
                    and m.ratio() >= SIMILAR_RATIO:
                pairs.append(f"{a!r} ~ {b!r}")
    return _issue("warning", "similar_labels",
                  "Labels that are nearly identical. If they mean the same, classifiers "
                  "are penalised for choosing the 'wrong' twin; consider merging them.",
                  pairs)


# --- gold standard ----------------------------------------------------------

def _check_gold(task: Task) -> list[Issue]:
    id_col, label_col = task.id_col, task.label_col
    rows = task.rows
    ids = Counter(r[id_col] for r in rows)
    issues = _issue("error", "duplicate_id", "Ids that appear more than once in gold.csv.",
                    [f"{i!r} ({c}x)" for i, c in ids.items() if c > 1])
    issues += _issue("error", "empty_gold_label", "Items without a gold label.",
                     [r[id_col] for r in rows if not r[label_col].strip()])

    known = sorted(task.label_set)
    stripped = {n.strip(): n for n in known}
    unknown = []
    for r in rows:
        lab = r[label_col]
        if not lab.strip() or lab in task.label_set:
            continue
        if lab.strip() in stripped:
            hint = f"matches {stripped[lab.strip()]!r} after removing surrounding whitespace"
        else:
            close = get_close_matches(lab, known, n=3, cutoff=0.6)
            hint = ("did you mean " + ", ".join(repr(c) for c in close)) if close else "no similar label"
        unknown.append(f"id {r[id_col]}: {lab!r} ({hint})")
    issues += _issue("error", "unknown_gold_label",
                     "Gold labels that are not part of labels.csv.", unknown)

    if task.weight_col:
        bad = []
        for r in rows:
            try:
                if float(r[task.weight_col]) < 0:
                    bad.append(f"id {r[id_col]}: {r[task.weight_col]!r}")
            except ValueError:
                bad.append(f"id {r[id_col]}: {r[task.weight_col]!r}")
        issues += _issue("error", "bad_weight", "Weights that are not non-negative numbers.", bad)

    if task.split_col:
        issues += _issue("warning", "unknown_split",
                         f"Values in {task.split_col!r} other than dev or test; these items "
                         "are never evaluated by default.",
                         [f"id {r[id_col]}: {r[task.split_col]!r}" for r in rows
                          if r[task.split_col] not in KNOWN_SPLITS])

    issues += _issue("info", "empty_features", "Items whose feature columns are all empty.",
                     [r[id_col] for r in rows if not any(r[f].strip() for f in task.features)])
    label_counts = Counter(r[label_col] for r in rows)
    issues += _issue("info", "singleton_classes",
                     "Labels with exactly one gold example; per-class numbers for them "
                     "are anecdotal.",
                     sorted(lab for lab, c in label_counts.items() if c == 1))
    return issues


# --- hierarchy levels -------------------------------------------------------

def _check_levels(task: Task) -> list[Issue]:
    issues: list[Issue] = []
    for level in task.levels:
        issues += _issue("warning", "empty_level_value",
                         f"Labels without a value for level {level!r}.",
                         [r["label"] for r in task.labels if not r[level].strip()])

    for level, column in task.level_gold.items():
        mapping = task.level_map(level)
        values = set(mapping.values())
        issues += _issue("warning", "unknown_level_value",
                         f"Values in {column!r} that do not occur in level {level!r} "
                         "of labels.csv.",
                         [f"id {r[task.id_col]}: {r[column]!r}" for r in task.rows
                          if r[column] not in values])
        mismatch = [
            f"id {r[task.id_col]}: {r[task.label_col]!r} is {mapping[r[task.label_col]]!r} "
            f"in labels.csv, but {r[column]!r} in {column!r}"
            for r in task.rows
            if r[task.label_col] in mapping and r[column] != mapping[r[task.label_col]]
        ]
        issues += _issue("warning", "level_gold_mismatch",
                         f"Items whose own {level!r} differs from the {level!r} of their "
                         "gold label. Evaluation on this level uses the item's own value.",
                         mismatch)
    return issues


# --- duplicate inputs -------------------------------------------------------

def _check_duplicates(task: Task) -> list[Issue]:
    groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for r in task.rows:
        key = tuple(" ".join(r[f].split()) for f in task.features)
        if any(key):
            groups[key].append(r)

    conflicting, repeated = [], 0
    for rows in groups.values():
        if len(rows) < 2:
            continue
        labels = Counter(r[task.label_col] for r in rows)
        if len(labels) > 1:
            ids = ", ".join(r[task.id_col] for r in rows)
            shown = "; ".join(f"{lab!r} ({c}x)" for lab, c in labels.most_common())
            conflicting.append(f"ids {ids}: {shown}")
        else:
            repeated += len(rows) - 1

    issues = _issue("warning", "conflicting_duplicates",
                    "Items with identical features but different gold labels. "
                    "No classifier can get all of them right.", conflicting)
    if repeated:
        issues.append(Issue("info", "repeated_items",
                            f"{repeated} items repeat the features and label of another "
                            "item; they weigh the same input more than once.", repeated))
    return issues
