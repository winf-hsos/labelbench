"""End-to-end tests on a small toy task."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from labelbench.checks import run_checks
from labelbench.compare import compare
from labelbench.io import read_csv, read_json
from labelbench.metrics import evaluate, mcnemar_exact
from labelbench.runner import run
from labelbench.task import INVALID, TaskError, load_task

LABELS = """\
label,description,diet
spaghetti_bolognese,Pasta with meat sauce,meat
lentil_bolognese,Pasta with lentil sauce,vegan
fish_and_chips,Fried fish with fries,fish
cheese_spaetzle,Spaetzle with cheese,vegetarian
"""

GOLD = '''\
id,name,notes,label,split,diet_item
1,Spaghetti Bolognese,beef,spaghetti_bolognese,dev,meat
2,Linsen-Bolognese,vegan,lentil_bolognese,dev,vegan
3,Backfisch mit Pommes,fish,fish_and_chips,dev,fish
4,Käsespätzle,"milk, egg",cheese_spaetzle,dev,vegetarian
5,Spaghetti Bolognese,beef,spaghetti_bolognese,test,meat
6,"Fischstäbchen, ""knusprig""","fish
breaded",fish_and_chips,dev,fish
'''

CLASSIFIER = '''\
from labelbench import Prediction


class Keywords:
    def __init__(self, fallback="cheese_spaetzle"):
        self.fallback = fallback
        self.labels = None

    def prepare(self, task):
        self.labels = [row["label"] for row in task.labels]

    def predict(self, items):
        out = []
        for item in items:
            text = (item["name"] + " " + item["notes"]).lower()
            if "linsen" in text:
                label = "lentil_bolognese"
            elif "bolognese" in text:
                label = "spaghetti_bolognese"
            elif "fisch" in text:
                label = "fish_and_chips"
            else:
                label = self.fallback
            out.append(Prediction(label=label, raw=text, candidates=self.labels))
        return out
'''


def write_task(root: Path, task_yaml: str, gold: str = GOLD, labels: str = LABELS) -> Path:
    task = root / "task"
    task.mkdir(parents=True, exist_ok=True)
    (task / "task.yaml").write_text(textwrap.dedent(task_yaml), encoding="utf-8")
    (task / "labels.csv").write_text(labels, encoding="utf-8", newline="")
    (task / "gold.csv").write_text(gold, encoding="utf-8", newline="")
    return task


TASK_YAML = """\
    name: toy
    type: single
    id: id
    features: [name, notes]
    label: label
    labels: labels.csv
    split: split
    levels: [diet]
    level_gold: {diet: diet_item}
    language: de
"""


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "toy_clf.py").write_text(CLASSIFIER, encoding="utf-8")
    (tmp_path / "good.yaml").write_text("classifier: toy_clf:Keywords\n", encoding="utf-8")
    (tmp_path / "worse.yaml").write_text(
        "classifier: toy_clf:Keywords\nlabel_map: {lentil_bolognese: nonsense}\n",
        encoding="utf-8")
    write_task(tmp_path, TASK_YAML)
    return tmp_path


def test_load_task_reads_multiline_fields(project):
    task = load_task(project / "task")
    assert len(task.rows) == 6
    assert task.rows[5]["notes"] == "fish\nbreaded"
    assert [r["id"] for r in task.select("dev")] == ["1", "2", "3", "4", "6"]


def test_run_writes_complete_run_directory(project):
    run_dir = run(project / "task", project / "good.yaml")
    for name in ("task.yaml", "labels.csv", "gold.csv", "config.yaml", "predictions.csv",
                 "checks.json", "provenance.json", "metrics.json", "report.html"):
        assert (run_dir / name).is_file(), name
    metrics = read_json(run_dir / "metrics.json")
    assert metrics["split"] == "dev"
    assert metrics["n_items"] == 5
    assert metrics["levels"]["label"]["accuracy"] == 1.0
    assert metrics["levels"]["diet"]["accuracy"] == 1.0
    assert metrics["levels"]["label"]["top_k"]["1"] == 1.0
    html = (run_dir / "report.html").read_text(encoding="utf-8")
    assert "__LABELBENCH_DATA__" not in html
    assert "</script>" not in html.split('id="data">')[1].split("</script>")[0]


def test_test_split_is_logged(project):
    run(project / "task", project / "good.yaml", split="test")
    log = (project / "runs" / "test_access.log").read_text(encoding="utf-8")
    assert "good.yaml" in log


def test_invalid_predictions_are_counted(project):
    run_dir = run(project / "task", project / "worse.yaml")
    preds = read_csv(run_dir / "predictions.csv")
    mapped = [p for p in preds if p["label"] == "nonsense"]
    assert len(mapped) == 1
    assert json.loads(mapped[0]["meta"])["label_before_map"] == "lentil_bolognese"
    m = read_json(run_dir / "metrics.json")["levels"]["label"]
    assert m["accuracy"] == pytest.approx(0.8)
    assert m["invalid_share"] == pytest.approx(0.2)


def test_compare_finds_broken_item(project):
    a = run(project / "task", project / "good.yaml")
    b = run(project / "task", project / "worse.yaml")
    out = compare(a, b)
    doc = read_json(out / "comparison.json")
    assert doc["levels"]["label"]["broken"] == 1
    assert doc["levels"]["label"]["fixed"] == 0
    assert doc["warnings"] == []
    assert (out / "report.html").is_file()


def test_metrics_and_level_gold():
    gold = ["a", "a", "b", "c"]
    pred = ["a", "b", "b", INVALID]
    maps = {"group": {"a": "x", "b": "x", "c": "y"}}
    res = evaluate(gold, pred, level_maps=maps, level_gold={"group": ["x", "x", "x", "y"]})
    assert res["label"]["accuracy"] == 0.5
    assert res["label"]["invalid_share"] == 0.25
    assert res["group"]["accuracy"] == 0.75
    lo, hi = res["label"]["accuracy_ci"]
    assert 0 <= lo <= 0.5 <= hi <= 1


def test_mcnemar():
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(5, 5) == 1.0
    assert mcnemar_exact(10, 0) == pytest.approx(2 / 1024)


def test_checks_report_problems_instead_of_raising(project):
    gold = GOLD + "7,Pasta,,Spaghetti_Bolognese ,dev,meat\n" + \
        "8,Käsespätzle,\"milk, egg\",spaghetti_bolognese,dev,vegetarian\n" + "3,dup,,fish_and_chips,dev,fish\n"
    labels = LABELS + "lentil-bolognese,dupe,vegan\n"
    write_task(project, TASK_YAML, gold=gold, labels=labels)
    with pytest.raises(TaskError):
        load_task(project / "task")
    task = load_task(project / "task", strict=False)
    codes = {i.code: i for i in task.issues}
    assert "unknown_gold_label" in codes
    assert "did you mean" in codes["unknown_gold_label"].examples[0]
    assert "duplicate_id" in codes
    assert "similar_labels" in codes
    assert "conflicting_duplicates" in codes
    assert "level_gold_mismatch" in codes
    assert [i.severity for i in task.issues] == sorted(
        [i.severity for i in task.issues], key=["error", "warning", "info"].index)
    assert run_checks(task) == task.issues
