"""Tests for project templates and the built-in LLM classifier (without network access)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from labelbench.classifier import TaskInfo, load_classifier
from labelbench.io import read_json, read_yaml
from labelbench.providers.anthropic import FALLBACK_BETA, AnthropicProvider
from labelbench.runner import run
from labelbench.scaffold import PROVIDERS, ScaffoldError, init_project, new_classifier, new_task
from labelbench.task import load_task

FAKE_PROVIDER = '''\
import json


def complete(system, user, schema, model, **options):
    labels = schema["properties"]["label"]["enum"]
    text = user.lower()
    label = next((l for l in labels if l.split("_")[0] in text), labels[0])
    return {"text": json.dumps({"reasoning": "fake", "label": label}),
            "usage": {"input_tokens": len(system or "") + len(user)}, "model": model}
'''


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    init_project(".")
    return tmp_path


def test_init_creates_runnable_example(project):
    assert (project / ".gitignore").is_file()
    assert load_task(project / "tasks" / "example").issues == []
    run_dir = run("tasks/example", "configs/keywords-v1.yaml")
    metrics = read_json(run_dir / "metrics.json")
    assert metrics["n_items"] == 16
    assert metrics["levels"]["label"]["accuracy"] > 0.5


def test_init_never_overwrites(project):
    with pytest.raises(ScaffoldError, match="already exist"):
        init_project(".")


def test_new_task_and_rules_classifier_use_task_labels(project):
    new_task(".", "tickets")
    spec = read_yaml(project / "tasks" / "tickets" / "task.yaml")
    assert spec["name"] == "tickets"
    new_classifier(".", "my-rules", "rules", task="tasks/example")
    config = read_yaml(project / "configs" / "my-rules-v1.yaml")
    assert config["classifier"] == "classifiers.my_rules:KeywordClassifier"
    assert set(config["params"]["keywords"]) == {
        r["label"] for r in load_task(project / "tasks" / "example").labels}


@pytest.mark.parametrize("provider", PROVIDERS)
def test_llm_templates_are_valid_yaml(project, provider):
    new_classifier(".", f"llm-{provider}", "llm", provider=provider, task="tasks/example")
    config = read_yaml(project / "configs" / f"llm-{provider}-v1.yaml")
    assert config["classifier"] == "llm"
    assert config["params"]["prompt"].endswith("-item.md")
    item = (project / config["params"]["prompt"]).read_text(encoding="utf-8")
    assert "{{subject}}" in item and "{{message}}" in item
    if provider == "custom":
        assert (project / "classifiers" / f"llm_{provider}_provider.py").is_file()


def test_invalid_names_are_rejected(project):
    with pytest.raises(ScaffoldError):
        new_task(".", "Bad Name")


def test_llm_classifier_with_custom_provider_and_cache(project):
    from labelbench.io import read_csv

    new_classifier(".", "fake", "llm", provider="custom", task="tasks/example")
    # Replace the generated stub with a provider that answers without network access.
    (project / "classifiers" / "fake_provider.py").write_text(FAKE_PROVIDER, encoding="utf-8")

    first = run("tasks/example", "configs/fake-v1.yaml")
    second = run("tasks/example", "configs/fake-v1.yaml")
    meta1 = [json.loads(r["meta"]) for r in read_csv(first / "predictions.csv")]
    meta2 = [json.loads(r["meta"]) for r in read_csv(second / "predictions.csv")]
    assert not any(m["cached"] for m in meta1)
    assert all(m["cached"] for m in meta2)
    assert all(m["enum"] for m in meta1)
    assert read_json(first / "metrics.json") == read_json(second / "metrics.json")


def test_unknown_placeholder_is_reported(project):
    new_classifier(".", "typo", "llm", provider="openai", task="tasks/example")
    item = project / "prompts" / "typo-v1-item.md"
    item.write_text("Text: {{body}}\n", encoding="utf-8")
    from labelbench.llm import LLMClassifier

    clf = LLMClassifier.__new__(LLMClassifier)
    clf.item_template = item.read_text(encoding="utf-8")
    clf.system_template = None
    clf.label_line = "- {label}"
    with pytest.raises(ValueError, match="body"):
        clf.prepare(TaskInfo(name="t", type="single", features=["subject"],
                             labels=[{"label": "a"}], levels=[]))


class _FakeMessages:
    def __init__(self, stop_reason="end_turn"):
        self.calls = []
        self.stop_reason = stop_reason

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            stop_reason=self.stop_reason, stop_details=None, model="served-model",
            content=[SimpleNamespace(type="thinking", thinking=""),
                     SimpleNamespace(type="text", text='{"label": "a"}')],
            usage=SimpleNamespace(input_tokens=10, output_tokens=3,
                                  cache_read_input_tokens=8, cache_creation_input_tokens=0),
        )


def _anthropic(monkeypatch, **options):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    provider = AnthropicProvider(model="claude-opus-5-5", **options)
    fake, fake_beta = _FakeMessages(), _FakeMessages()
    provider.client = SimpleNamespace(messages=fake, beta=SimpleNamespace(messages=fake_beta))
    return provider, fake, fake_beta


def test_anthropic_request_shape(monkeypatch):
    provider, fake, fake_beta = _anthropic(monkeypatch, effort="low")
    schema = {"type": "object", "properties": {"label": {"type": "string", "enum": ["a"]}},
              "required": ["label"], "additionalProperties": False}
    result = provider.complete("system text", "user text", schema)
    call = fake.calls[0]
    assert call["output_config"] == {"format": {"type": "json_schema", "schema": schema},
                                     "effort": "low"}
    assert call["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert call["messages"] == [{"role": "user", "content": "user text"}]
    assert result["text"] == '{"label": "a"}'
    assert result["model"] == "served-model"
    assert result["usage"]["cache_read_input_tokens"] == 8
    assert fake_beta.calls == []


def test_anthropic_fallbacks_use_beta_endpoint(monkeypatch):
    provider, fake, fake_beta = _anthropic(monkeypatch, fallbacks="default")
    provider.complete(None, "user text", {"type": "object"})
    assert fake.calls == []
    assert fake_beta.calls[0]["betas"] == [FALLBACK_BETA]
    assert fake_beta.calls[0]["fallbacks"] == "default"
    assert "system" not in fake_beta.calls[0]


def test_anthropic_refusal_raises(monkeypatch):
    provider, fake, _ = _anthropic(monkeypatch)
    fake.stop_reason = "refusal"
    with pytest.raises(RuntimeError, match="declined"):
        provider.complete(None, "user text", {"type": "object"})


def test_builtin_llm_is_registered(project, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    new_classifier(".", "oa", "llm", provider="openai", task="tasks/example")
    clf, config = load_classifier("configs/oa-v1.yaml")
    assert type(clf).__name__ == "LLMClassifier"
    assert config["params"]["model"] == "gpt-6-luna"
