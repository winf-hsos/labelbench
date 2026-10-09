"""Running a classifier on a task, writing the run directory and evaluating it.

A run directory is self-contained: it holds copies of the task description,
the label schema and the evaluated gold rows next to the predictions, so it
can be re-evaluated, re-reported and compared without the original files.
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from labelbench import __version__
from labelbench.checks import Issue
from labelbench.classifier import Prediction, TaskInfo, load_classifier
from labelbench.io import (
    csv_columns,
    read_csv,
    read_json,
    read_yaml,
    sha256_file,
    write_csv,
    write_json,
    write_text_atomic,
    write_yaml,
)
from labelbench.metrics import evaluate
from labelbench.report import render_run_report, run_report_data
from labelbench.task import INVALID, Task, TaskError, load_task

PREDICTION_COLUMNS = ["id", "label", "candidates", "raw", "meta"]


def run(task_path: str | Path, clf_path: str | Path, *, split: str | None = None,
        runs_dir: str | Path = "runs") -> Path:
    task = load_task(task_path)
    if split is None and task.split_col:
        split = "dev"
    rows = task.select(split)

    classifier, config = load_classifier(clf_path)
    if hasattr(classifier, "prepare"):
        classifier.prepare(TaskInfo(name=task.name, type=task.type, features=task.features,
                                    labels=task.labels, levels=task.levels))
    items = task.items(rows)
    started = time.perf_counter()
    predictions = _predict(classifier, items, config.get("batch_size"))
    runtime = time.perf_counter() - started

    run_dir = _new_dir(Path(runs_dir), f"{task.name}_{Path(clf_path).stem}")
    _copy_task(task, rows, run_dir)
    write_yaml(run_dir / "config.yaml", config)
    write_csv(run_dir / "predictions.csv",
              _prediction_rows(items, predictions, config.get("label_map") or {}),
              PREDICTION_COLUMNS)
    write_json(run_dir / "checks.json", [i.as_dict() for i in task.issues])
    write_json(run_dir / "provenance.json",
               _provenance(task, split, len(rows), clf_path, config, runtime))
    if split == "test":
        _log_test_access(Path(runs_dir), task, clf_path, run_dir)

    evaluate_run_dir(run_dir)
    return run_dir


def predict(task_path: str | Path, items_path: str | Path, clf_path: str | Path,
            out_dir: str | Path) -> Path:
    """Classify unlabelled items with exactly the classifier used in evaluation."""
    task = load_task(task_path, strict=False)
    columns = csv_columns(items_path)
    absent = [c for c in [task.id_col, *task.features] if c not in columns]
    if absent:
        raise TaskError(f"Columns {absent} are missing in {items_path}.")
    rows = read_csv(items_path)
    classifier, config = load_classifier(clf_path)
    if hasattr(classifier, "prepare"):
        classifier.prepare(TaskInfo(name=task.name, type=task.type, features=task.features,
                                    labels=task.labels, levels=task.levels))
    items = task.items(rows)
    started = time.perf_counter()
    predictions = _predict(classifier, items, config.get("batch_size"))
    runtime = time.perf_counter() - started

    out = _new_dir(Path(out_dir), f"predict_{task.name}_{Path(clf_path).stem}")
    write_yaml(out / "config.yaml", config)
    write_csv(out / "predictions.csv",
              _prediction_rows(items, predictions, config.get("label_map") or {}),
              PREDICTION_COLUMNS)
    prov = _provenance(task, None, len(rows), clf_path, config, runtime)
    prov["items"] = {"path": str(Path(items_path).resolve()), "sha256": sha256_file(items_path)}
    write_json(out / "provenance.json", prov)
    return out


def evaluate_run_dir(run_dir: str | Path) -> dict[str, Any]:
    """Compute metrics.json and report.html from the files of a run directory."""
    run_dir = Path(run_dir)
    task = load_task(run_dir, strict=False)
    rows = task.rows
    pred_rows = load_predictions(run_dir, [r[task.id_col] for r in rows])

    gold = [r[task.label_col] for r in rows]
    pred = [normalise(p["label"], task.label_set) for p in pred_rows]
    candidates = None
    if any(p["candidates"] for p in pred_rows):
        candidates = [[c for c in _json_list(p["candidates"]) if c in task.label_set]
                      for p in pred_rows]
    weights = [float(r[task.weight_col]) for r in rows] if task.weight_col else None
    level_maps = {lv: task.level_map(lv) for lv in task.levels}
    level_gold = {lv: [r[col] for r in rows] for lv, col in task.level_gold.items()}

    provenance = read_json(run_dir / "provenance.json")
    metrics = {
        "task": task.name,
        "split": provenance["task"].get("split"),
        "n_items": len(rows),
        "levels": evaluate(gold, pred, level_maps=level_maps, level_gold=level_gold,
                           candidates=candidates, weights=weights),
    }
    write_json(run_dir / "metrics.json", metrics)

    checks_file = run_dir / "checks.json"
    issues = [Issue(**i) for i in read_json(checks_file)] if checks_file.is_file() else []
    data = run_report_data(
        run_name=run_dir.name, task=task, rows=rows, pred_rows=pred_rows, pred=pred,
        metrics=metrics, provenance=provenance, config=read_yaml(run_dir / "config.yaml"),
        issues=issues,
    )
    write_text_atomic(run_dir / "report.html", render_run_report(data))
    return metrics


def load_predictions(run_dir: Path, ids: list[str]) -> list[dict[str, str]]:
    """Predictions in the order of `ids`; every id must occur exactly once."""
    rows = read_csv(run_dir / "predictions.csv")
    by_id: dict[str, dict[str, str]] = {}
    for r in rows:
        if r["id"] in by_id:
            raise TaskError(f"Id {r['id']} occurs twice in predictions.csv.")
        by_id[r["id"]] = r
    missing = [i for i in ids if i not in by_id]
    if missing:
        raise TaskError(f"{len(missing)} items have no prediction, e.g. {missing[:5]}.")
    extra = set(by_id) - set(ids)
    if extra:
        raise TaskError(f"{len(extra)} predictions refer to unknown ids, e.g. {sorted(extra)[:5]}.")
    return [by_id[i] for i in ids]


def normalise(label: str, label_set: set[str]) -> str:
    return label if label in label_set else INVALID


def _predict(classifier: Any, items: list[dict[str, str]], batch_size: int | None) -> list[Prediction]:
    size = batch_size or len(items) or 1
    out: list[Prediction] = []
    for start in range(0, len(items), size):
        batch = items[start:start + size]
        result = list(classifier.predict(batch))
        if len(result) != len(batch):
            raise RuntimeError(f"Classifier returned {len(result)} predictions for {len(batch)} items.")
        out += result
        if size < len(items):
            print(f"  {len(out)}/{len(items)} items classified", file=sys.stderr)
    return out


def _prediction_rows(items: list[dict[str, str]], predictions: list[Prediction],
                     label_map: dict[str, str]) -> list[dict[str, str]]:
    rows = []
    for item, p in zip(items, predictions, strict=True):
        meta = dict(p.meta)
        label = p.label
        if label is not None and label in label_map:
            meta["label_before_map"] = label
            label = label_map[label]
        candidates = [label_map.get(c, c) for c in p.candidates] if p.candidates else None
        rows.append({
            "id": item["id"],
            "label": "" if label is None else str(label),
            "candidates": json.dumps(candidates, ensure_ascii=False) if candidates else "",
            "raw": p.raw,
            "meta": json.dumps(meta, ensure_ascii=False) if meta else "",
        })
    return rows


def _json_list(text: str) -> list[str]:
    if not text:
        return []
    value = json.loads(text)
    return [str(v) for v in value] if isinstance(value, list) else []


def _new_dir(parent: Path, name: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    parent.mkdir(parents=True, exist_ok=True)
    candidate = parent / f"{stamp}_{name}"
    n = 2
    while candidate.exists():
        candidate = parent / f"{stamp}_{name}_{n}"
        n += 1
    candidate.mkdir()
    return candidate


def _copy_task(task: Task, rows: list[dict[str, str]], run_dir: Path) -> None:
    spec = dict(task.spec)
    spec["labels"] = "labels.csv"
    spec["gold"] = "gold.csv"
    write_yaml(run_dir / "task.yaml", spec)
    shutil.copyfile(task.labels_file, run_dir / "labels.csv")
    write_csv(run_dir / "gold.csv", rows, csv_columns(task.gold_file))


def _provenance(task: Task, split: str | None, n_items: int, clf_path: str | Path,
                config: dict[str, Any], runtime: float) -> dict[str, Any]:
    files = {}
    for key, value in (config.get("params") or {}).items():
        if isinstance(value, str) and Path(value).is_file():
            files[key] = {"path": value, "sha256": sha256_file(value)}
    return {
        "labelbench_version": __version__,
        "created": datetime.now().astimezone().isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "task": {
            "path": str(task.root.resolve()),
            "name": task.name,
            "split": split,
            "n_items": n_items,
            "sha256": {
                "task.yaml": sha256_file(task.root / "task.yaml"),
                "labels.csv": sha256_file(task.labels_file),
                "gold.csv": sha256_file(task.gold_file),
            },
        },
        "classifier": {
            "config": str(Path(clf_path).resolve()),
            "config_sha256": sha256_file(clf_path),
            "classifier": config.get("classifier"),
            "files": files,
        },
        "runtime_seconds": round(runtime, 2),
    }


def _log_test_access(runs_dir: Path, task: Task, clf_path: str | Path, run_dir: Path) -> None:
    line = "\t".join([datetime.now().astimezone().isoformat(timespec="seconds"),
                      str(task.root.resolve()), str(Path(clf_path).resolve()), run_dir.name])
    with (runs_dir / "test_access.log").open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")
