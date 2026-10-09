"""Building the self-contained HTML report.

The report is one HTML file with the data embedded as JSON and all styling
and logic inline, so it needs no server and can be sent by e-mail. The page
itself lives in report_template.html; this module only assembles the data.
"""

from __future__ import annotations

import json
from importlib import resources
from typing import TYPE_CHECKING, Any

from labelbench.checks import Issue

if TYPE_CHECKING:
    from labelbench.task import Task

PLACEHOLDER = "__LABELBENCH_DATA__"


def run_report_data(*, run_name: str, task: Task, rows: list[dict[str, str]],
                    pred_rows: list[dict[str, str]], pred: list[str], metrics: dict[str, Any],
                    provenance: dict[str, Any], config: dict[str, Any],
                    issues: list[Issue]) -> dict[str, Any]:
    from labelbench.task import INVALID

    level_maps = {lv: task.level_map(lv) for lv in task.levels}
    items = []
    for r, p, label in zip(rows, pred_rows, pred, strict=True):
        gold = r[task.label_col]
        levels = {"label": {"gold": gold, "pred": label}}
        for lv, mapping in level_maps.items():
            col = task.level_gold.get(lv)
            levels[lv] = {"gold": r[col] if col else mapping[gold],
                          "pred": mapping.get(label, INVALID)}
        items.append({
            "id": r[task.id_col],
            "features": {f: r[f] for f in task.features},
            "show": {c: r[c] for c in task.show},
            "gold": gold,
            "pred": label,
            "raw_label": p["label"],
            "probability": float(p["probability"]) if p.get("probability") else None,
            "candidates": _loads(p.get("candidates", ""), []),
            "raw": p.get("raw", ""),
            "meta": _loads(p.get("meta", ""), {}),
            "levels": levels,
        })
    return {
        "kind": "run",
        "language": task.spec.get("language", "en"),
        "has_probability": bool(config.get("probability")),
        "title": task.name,
        "run_name": run_name,
        "task": _task_info(task),
        "labels": task.labels,
        "metrics": metrics,
        "provenance": provenance,
        "config": config,
        "issues": [i.as_dict() for i in issues],
        "items": items,
    }


def render_run_report(data: dict[str, Any]) -> str:
    return _render(data)


def render_compare_report(data: dict[str, Any]) -> str:
    return _render(data)


def _task_info(task: Task) -> dict[str, Any]:
    return {
        "name": task.name,
        "description": task.spec.get("description", ""),
        "features": task.features,
        "show": task.show,
        "levels": task.levels,
        "level_gold": task.level_gold,
        "weight": task.weight_col,
        "n_labels": len(task.labels),
    }


def _render(data: dict[str, Any]) -> str:
    template = resources.files("labelbench").joinpath("report_template.html").read_text("utf-8")
    # "<" is escaped so that no text in the data can close the script element.
    payload = json.dumps(data, ensure_ascii=False).replace("<", "\\u003c")
    return template.replace(PLACEHOLDER, payload)


def _loads(text: str, default: Any) -> Any:
    if not text:
        return default
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return default
