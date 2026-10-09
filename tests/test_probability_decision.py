"""Tests for probabilities and the Decisions API classifier (without network access)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from labelbench.classifier import ClassifierError, TaskInfo
from labelbench.decision import NONE, DecisionClassifier, combine
from labelbench.io import read_csv, read_json
from labelbench.metrics import probability_metrics
from labelbench.runner import run
from labelbench.scaffold import init_project, new_classifier

PROB_CLASSIFIER = '''\
from labelbench import Prediction


class Sure:
    def __init__(self):
        self.on = False

    def enable_probability(self):
        self.on = True

    def predict(self, items):
        out = []
        for i, item in enumerate(items):
            label = "refund_request" if "money" in item["message"] else "bug_report"
            out.append(Prediction(label=label, probability=0.9 if i % 2 else 0.6))
        return out


class NoProb:
    def predict(self, items):
        return [Prediction(label="bug_report") for _ in items]
'''

FAKE_PROVIDER = '''\
import json


def complete(system, user, schema, model, **options):
    labels = schema["properties"]["label"]["enum"]
    answer = {"reasoning": "fake", "label": labels[0]}
    if "probability" in schema["properties"]:
        answer["probability"] = 1.7
    return json.dumps(answer)
'''


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    init_project(".")
    (tmp_path / "classifiers" / "probs.py").write_text(PROB_CLASSIFIER, encoding="utf-8")
    return tmp_path


def test_probability_metrics():
    m = probability_metrics([True, False, True, True, True], [0.95, 0.6, 0.9, None, 0.55])
    assert m["n"] == 4 and m["n_missing"] == 1
    assert m["accuracy"] == pytest.approx(0.75)
    assert sum(b["n"] for b in m["bins"]) == 4
    cov = {c["threshold"]: c for c in m["coverage"]}
    assert cov[0.9]["n"] == 2 and cov[0.9]["accuracy"] == 1.0
    assert 0 <= m["ece"] <= 1 and 0 <= m["brier"] <= 1
    # the wrong item (0.6) ranks above one correct item (0.55) and below two: AUC 2/3
    assert m["auc"] == pytest.approx(2 / 3)


def test_probability_is_reported_when_configured(project):
    (project / "configs" / "sure.yaml").write_text(
        "classifier: classifiers.probs:Sure\nprobability: true\n", encoding="utf-8")
    run_dir = run("tasks/example", "configs/sure.yaml")
    preds = read_csv(run_dir / "predictions.csv")
    assert {p["probability"] for p in preds} == {"0.6", "0.9"}
    m = read_json(run_dir / "metrics.json")["levels"]["label"]["probability"]
    assert m["n"] == 16 and m["n_missing"] == 0
    html = (run_dir / "report.html").read_text(encoding="utf-8")
    assert '"has_probability": true' in html


def test_probability_is_ignored_unless_configured(project):
    (project / "configs" / "sure.yaml").write_text("classifier: classifiers.probs:Sure\n", encoding="utf-8")
    run_dir = run("tasks/example", "configs/sure.yaml")
    assert {p["probability"] for p in read_csv(run_dir / "predictions.csv")} == {""}
    assert "probability" not in read_json(run_dir / "metrics.json")["levels"]["label"]


def test_probability_requires_capable_classifier(project):
    (project / "configs" / "noprob.yaml").write_text(
        "classifier: classifiers.probs:NoProb\nprobability: true\n", encoding="utf-8")
    with pytest.raises(ClassifierError, match="cannot deliver probabilities"):
        run("tasks/example", "configs/noprob.yaml")


def test_llm_self_reported_probability_is_clamped(project):
    new_classifier(".", "fake", "llm", provider="custom", task="tasks/example")
    (project / "classifiers" / "fake_provider.py").write_text(FAKE_PROVIDER, encoding="utf-8")
    cfg = project / "configs" / "fake-v1.yaml"
    cfg.write_text(cfg.read_text(encoding="utf-8") + "probability: true\n", encoding="utf-8")
    run_dir = run("tasks/example", "configs/fake-v1.yaml")
    preds = read_csv(run_dir / "predictions.csv")
    assert {p["probability"] for p in preds} == {"1"}
    assert json.loads(preds[0]["meta"])["probability_source"].startswith("self-reported")


def test_combine_single_and_split_parts():
    single = combine([["a", "b"]], [{"name": "part0", "probabilities": {"a": 0.6, "b": 0.2}}])
    assert single == pytest.approx({"a": 0.75, "b": 0.25})
    split = combine([["a", "b"], ["c"]], [
        {"name": "part0", "probabilities": {"a": 0.8, "b": 0.1, NONE: 0.1}},
        {"name": "part1", "probabilities": {"c": 0.1, NONE: 0.9}},
    ])
    assert max(split, key=split.get) == "a"
    assert sum(split.values()) == pytest.approx(1.0)
    assert split["a"] == pytest.approx(0.72 / (0.72 + 0.09 + 0.01))


def _answer(name, probs):
    best = max(probs, key=probs.get)
    return SimpleNamespace(type="choice", name=name, choice=best, confidence=probs[best],
                           probabilities=[SimpleNamespace(value=v, probability=p) for v, p in probs.items()])


def test_decision_classifier_splits_labels(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    (tmp_path / "q.md").write_text("Which label fits?", encoding="utf-8")
    (tmp_path / "item.md").write_text("Text: {{text}}", encoding="utf-8")
    clf = DecisionClassifier(instructions=str(tmp_path / "q.md"), prompt=str(tmp_path / "item.md"),
                             max_choices=3, top_k=2, cache_dir=str(tmp_path / "cache"))
    clf.enable_probability()
    labels = [{"label": x} for x in ("a", "b", "c", "d", "e")]
    clf.prepare(TaskInfo(name="t", type="single", features=["text"], labels=labels, levels=[]))
    assert [len(q["choices"]) for q in clf.questions] == [3, 3, 2]
    assert all(q["choices"][-1]["value"] == NONE for q in clf.questions)

    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(model="gpt-6-luna", usage=SimpleNamespace(input_tokens=10, output_tokens=0),
                               answers=[_answer("part0", {"a": 0.05, "b": 0.05, NONE: 0.9}),
                                        _answer("part1", {"c": 0.85, "d": 0.05, NONE: 0.1}),
                                        _answer("part2", {"e": 0.02, NONE: 0.98})])

    clf.client = SimpleNamespace(decisions=SimpleNamespace(create=create))
    first = clf.predict([{"id": "1", "text": "hello"}])[0]
    assert first.label == "c" and first.probability > 0.9
    assert first.candidates[0] == "c" and len(first.candidates) == 2
    assert calls[0]["input"] == "Text: hello"
    again = clf.predict([{"id": "1", "text": "hello"}])[0]
    assert len(calls) == 1 and again.meta["cached"] is True


def test_decision_refusal_counts_as_invalid(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    (tmp_path / "q.md").write_text("Which label fits?", encoding="utf-8")
    (tmp_path / "item.md").write_text("{{text}}", encoding="utf-8")
    clf = DecisionClassifier(instructions=str(tmp_path / "q.md"), prompt=str(tmp_path / "item.md"),
                             cache_dir=str(tmp_path / "cache"))
    clf.prepare(TaskInfo(name="t", type="single", features=["text"],
                         labels=[{"label": "a"}, {"label": "b"}], levels=[]))
    clf.client = SimpleNamespace(decisions=SimpleNamespace(create=lambda **kw: SimpleNamespace(
        model="gpt-6-luna", usage=SimpleNamespace(input_tokens=5, output_tokens=0),
        answers=[SimpleNamespace(type="refusal", name="part0")])))
    p = clf.predict([{"id": "1", "text": "x"}])[0]
    assert p.label is None and "refusal" in p.raw
