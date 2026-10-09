"""Comparing two runs item by item."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from labelbench.io import read_json, sha256_file, write_json, write_text_atomic
from labelbench.metrics import mcnemar_exact
from labelbench.report import render_compare_report
from labelbench.runner import load_predictions, normalise
from labelbench.task import INVALID, load_task


def compare(run_a: str | Path, run_b: str | Path, out_dir: str | Path | None = None) -> Path:
    a, b = _load(Path(run_a)), _load(Path(run_b))
    warnings = []
    if sha256_file(a["dir"] / "labels.csv") != sha256_file(b["dir"] / "labels.csv"):
        warnings.append("The two runs used different label schemas (labels.csv differs).")
    ids_a, ids_b = list(a["items"]), set(b["items"])
    common = [i for i in ids_a if i in ids_b]
    if len(common) != len(ids_a) or len(common) != len(ids_b):
        warnings.append(f"The runs cover different items; only the {len(common)} items in both "
                        "are compared.")
    changed_gold = [i for i in common if a["items"][i]["gold"] != b["items"][i]["gold"]]
    if changed_gold:
        warnings.append(f"{len(changed_gold)} items have different gold labels in the two runs, "
                        f"e.g. {changed_gold[:5]}; each run is scored against its own gold.")

    levels = ["label", *[lv for lv in a["task"].levels if lv in b["task"].levels]]
    summary = {}
    for level in levels:
        ca = [a["items"][i]["levels"][level]["correct"] for i in common]
        cb = [b["items"][i]["levels"][level]["correct"] for i in common]
        fixed = sum(1 for x, y in zip(ca, cb, strict=True) if not x and y)
        broken = sum(1 for x, y in zip(ca, cb, strict=True) if x and not y)
        n = len(common)
        summary[level] = {
            "n": n,
            "accuracy_a": sum(ca) / n if n else None,
            "accuracy_b": sum(cb) / n if n else None,
            "fixed": fixed,
            "broken": broken,
            "both_right": sum(1 for x, y in zip(ca, cb, strict=True) if x and y),
            "both_wrong": sum(1 for x, y in zip(ca, cb, strict=True) if not x and not y),
            "mcnemar_p": mcnemar_exact(fixed, broken),
            "macro_f1_a": a["metrics"]["levels"].get(level, {}).get("macro_f1"),
            "macro_f1_b": b["metrics"]["levels"].get(level, {}).get("macro_f1"),
        }

    task = b["task"]
    items = []
    for i in common:
        ia, ib = a["items"][i], b["items"][i]
        row = ib["row"]
        items.append({
            "id": i,
            "features": {f: row[f] for f in task.features},
            "show": {c: row.get(c, "") for c in task.show},
            "gold": ib["gold"],
            "a": {"pred": ia["pred"], "raw_label": ia["raw_label"], "raw": ia["raw"],
                  "levels": ia["levels"]},
            "b": {"pred": ib["pred"], "raw_label": ib["raw_label"], "raw": ib["raw"],
                  "levels": ib["levels"]},
        })

    parent = Path(out_dir) if out_dir else Path(run_a).parent
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = parent / f"{stamp}_compare"
    n = 2
    while out.exists():
        out = parent / f"{stamp}_compare_{n}"
        n += 1
    out.mkdir(parents=True)

    doc = {
        "created": datetime.now().astimezone().isoformat(timespec="seconds"),
        "run_a": str(a["dir"].resolve()),
        "run_b": str(b["dir"].resolve()),
        "warnings": warnings,
        "levels": summary,
    }
    write_json(out / "comparison.json", doc)
    data = {
        "kind": "compare",
        "language": task.spec.get("language", "en"),
        "title": f"{a['dir'].name}  vs.  {b['dir'].name}",
        "task": {"name": task.name, "features": task.features, "show": task.show,
                 "levels": task.levels},
        "runs": {
            "a": {"name": a["dir"].name, "config": a["config_name"], "metrics": a["metrics"]},
            "b": {"name": b["dir"].name, "config": b["config_name"], "metrics": b["metrics"]},
        },
        "comparison": doc,
        "items": items,
    }
    write_text_atomic(out / "report.html", render_compare_report(data))
    return out


def _load(run_dir: Path) -> dict[str, Any]:
    task = load_task(run_dir, strict=False)
    rows = task.rows
    preds = load_predictions(run_dir, [r[task.id_col] for r in rows])
    level_maps = {lv: task.level_map(lv) for lv in task.levels}
    items = {}
    for r, p in zip(rows, preds, strict=True):
        gold = r[task.label_col]
        pred = normalise(p["label"], task.label_set)
        levels = {"label": {"gold": gold, "pred": pred, "correct": gold == pred}}
        for lv, mapping in level_maps.items():
            col = task.level_gold.get(lv)
            g = r[col] if col else mapping[gold]
            pv = mapping.get(pred, INVALID)
            levels[lv] = {"gold": g, "pred": pv, "correct": g == pv}
        items[r[task.id_col]] = {
            "row": r, "gold": gold, "pred": pred, "raw_label": p["label"], "raw": p["raw"],
            "levels": levels,
        }
    provenance = read_json(run_dir / "provenance.json")
    return {
        "dir": run_dir,
        "task": task,
        "items": items,
        "metrics": read_json(run_dir / "metrics.json"),
        "config_name": Path(provenance["classifier"]["config"]).name,
    }
