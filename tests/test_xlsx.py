"""Excel workbooks as task input."""

from __future__ import annotations

import datetime as dt

import openpyxl
import pytest

from labelbench.io import read_csv, read_json, read_table, table_columns
from labelbench.runner import run
from labelbench.scaffold import init_project, new_task
from labelbench.task import TaskError, load_task


def workbook(path, sheets):
    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    for title, rows in sheets.items():
        ws = wb.create_sheet(title)
        for r in rows:
            ws.append(r)
    wb.save(path)
    return path


def test_read_table_converts_cells(tmp_path):
    path = workbook(tmp_path / "t.xlsx", {
        "first": [["x"], ["ignored"]],
        "data": [
            ["id", "name", None, "when", "flag"],
            [32839, "Gulasch\nmit Brot", "unnamed column", dt.datetime(2026, 10, 9), True],
            [None, None, None, None, None],
            [7.5, " spaces kept ", None, dt.date(2026, 1, 2), None],
        ],
    })
    rows = read_table(f"{path}#data")
    assert table_columns(f"{path}#data") == ["id", "name", "when", "flag"]
    assert rows == [
        {"id": "32839", "name": "Gulasch\nmit Brot", "when": "2026-10-09", "flag": "TRUE"},
        {"id": "7.5", "name": " spaces kept ", "when": "2026-01-02", "flag": ""},
    ]
    assert read_table(path) == [{"x": "ignored"}]   # first sheet by default


def test_read_table_errors(tmp_path):
    path = workbook(tmp_path / "t.xlsx", {"s": [["a", "a"], [1, 2]]})
    with pytest.raises(ValueError, match="duplicate column"):
        read_table(path)
    with pytest.raises(ValueError, match="no sheet"):
        read_table(f"{path}#missing")
    with pytest.raises(ValueError, match="unsupported"):
        read_table(tmp_path / "t.txt")


def test_task_from_one_workbook_matches_csv(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    init_project(".")
    gold = read_csv("tasks/example/gold.csv")
    labels = read_csv("tasks/example/labels.csv")
    gcols = list(gold[0])
    xdir = tmp_path / "tasks" / "example-xlsx"
    xdir.mkdir()
    workbook(xdir / "data.xlsx", {
        "Fälle": [gcols] + [[r[c] for c in gcols] for r in gold],
        "Themen": [["Thema", "description", "department"]]
                  + [[r["label"], r["description"], r["department"]] for r in labels],
    })
    spec = (tmp_path / "tasks" / "example" / "task.yaml").read_text(encoding="utf-8")
    spec = spec.replace("name: example", "name: example-xlsx").replace("labels: labels.csv", "labels: data.xlsx#Themen")
    (xdir / "task.yaml").write_text(spec + "gold: data.xlsx#Fälle\nlabels_column: Thema\n", encoding="utf-8")

    task = load_task(xdir)
    assert task.issues == [] and len(task.rows) == 20
    assert task.labels[0]["label"] == task.labels[0]["Thema"]

    a = run("tasks/example", "configs/keywords-v1.yaml")
    b = run("tasks/example-xlsx", "configs/keywords-v1.yaml")
    ma = read_json(a / "metrics.json")["levels"]
    mb = read_json(b / "metrics.json")["levels"]
    assert ma["label"]["accuracy"] == mb["label"]["accuracy"]
    assert ma["department"]["accuracy"] == mb["department"]["accuracy"]
    # the run directory always holds CSV copies
    assert (b / "labels.csv").is_file() and (b / "gold.csv").is_file()
    assert load_task(b).labels_column == "label"
    prov = read_json(b / "provenance.json")["task"]
    assert prov["sources"] == {"labels": "data.xlsx#Themen", "gold": "data.xlsx#Fälle"}


def test_missing_label_column_is_explained(tmp_path):
    workbook(tmp_path / "d.xlsx", {"g": [["id", "text", "label"], [1, "x", "a"]], "l": [["Name"], ["a"]]})
    (tmp_path / "task.yaml").write_text(
        "name: t\ntype: single\nid: id\nfeatures: [text]\nlabel: label\n"
        "labels: d.xlsx#l\ngold: d.xlsx#g\n", encoding="utf-8")
    with pytest.raises(TaskError, match="labels_column"):
        load_task(tmp_path)


def test_new_task_as_workbook(tmp_path):
    files = new_task(tmp_path, "tickets", "xlsx")
    assert {f.name for f in files} == {"task.yaml", "tickets.xlsx"}
    task = load_task(tmp_path / "tasks" / "tickets", strict=False)
    assert task.rows == [] and task.labels == []
    assert table_columns(tmp_path / "tasks" / "tickets" / "tickets.xlsx#gold") == ["id", "text", "label"]
