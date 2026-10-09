"""Creating projects, tasks and classifiers from the templates in templates/.

Nothing is ever overwritten: if any target file exists, no file is written
and the conflicting paths are reported.
"""

from __future__ import annotations

import re
from importlib import resources
from pathlib import Path

from labelbench import __version__
from labelbench.io import csv_columns, read_yaml

TEMPLATES = ("rules", "llm")
PROVIDERS = ("openai", "anthropic", "openai-compatible", "custom")
DEFAULT_MODELS = {
    "openai": "gpt-6-luna",
    "anthropic": "claude-opus-5-5",
    "openai-compatible": "your-model-name",
    "custom": "your-model-name",
}
PROVIDER_OPTIONS = {
    "openai": [
        "api_key_env: OPENAI_API_KEY      # environment variable holding the key",
        "# reasoning_effort: low          # for reasoning models",
    ],
    "anthropic": [
        "effort: low                      # low | medium | high | xhigh | max",
        "max_tokens: 4096                 # output limit including thinking",
        "fallbacks: default               # re-run declined requests on a fallback model;",
        "                                 # supported by current Opus/Sonnet/Fable models only",
        "# api_key_env: ANTHROPIC_API_KEY # default: the SDK finds the key itself",
    ],
    "openai-compatible": [
        "base_url: http://localhost:11434/v1   # URL of your server's OpenAI-style API",
        "# api_key_env: MY_SERVER_KEY          # if the server needs a key",
        "# json_schema: false                  # if the server lacks structured output",
    ],
    "custom": [
        "{}                               # passed as keyword arguments to your function",
    ],
}
EXAMPLE_KEYWORDS = {
    "refund_request": ["refund", "money back", "cancel", "charged twice"],
    "invoice_question": ["invoice", "bill", "charge", "vat"],
    "login_problem": ["sign in", "log in", "password", "locked", "two-factor", "2fa"],
    "address_change": ["address", "moved"],
    "delivery_status": ["parcel", "delivered", "delivery", "tracking", "arrived"],
    "bug_report": ["crash", "does nothing", "broken", "wrong", "finds nothing"],
}
NAME = re.compile(r"^[a-z][a-z0-9_-]*$")


class ScaffoldError(ValueError):
    """A name is invalid, a template is unknown or a target file exists."""


def init_project(directory: str | Path, llm: str | None = None,
                 model: str | None = None) -> list[Path]:
    root = Path(directory)
    files = {
        root / "README.md": _template("project/README.md", project=root.resolve().name,
                                      version=__version__),
        root / ".gitignore": _template("project/gitignore.txt"),
    }
    for name in ("task.yaml", "labels.csv", "gold.csv"):
        files[root / "tasks" / "example" / name] = _template(f"example/{name}")
    example = _ExampleTask()
    files.update(_rules_files(root, "keywords", example, EXAMPLE_KEYWORDS))
    if llm:
        files.update(_llm_files(root, "llm", llm, model, example))
    return _write(files)


def new_task(root: str | Path, name: str) -> list[Path]:
    _check_name(name)
    task_dir = Path(root) / "tasks" / name
    files = {task_dir / "task.yaml": _template("task/task.yaml", name=name,
                                               path=f"tasks/{name}")}
    files[task_dir / "labels.csv"] = _template("task/labels.csv")
    files[task_dir / "gold.csv"] = _template("task/gold.csv")
    return _write(files)


def new_classifier(root: str | Path, name: str, template: str, provider: str | None = None,
                   model: str | None = None, task: str | Path | None = None) -> list[Path]:
    _check_name(name)
    if template not in TEMPLATES:
        raise ScaffoldError(f"Unknown template {template!r}; available: {', '.join(TEMPLATES)}.")
    info = _TaskInfo(Path(task)) if task else None
    if template == "rules":
        return _write(_rules_files(Path(root), name, info, None))
    if not provider:
        raise ScaffoldError(f"The llm template needs --provider: {', '.join(PROVIDERS)}.")
    return _write(_llm_files(Path(root), name, provider, model, info))


def describe_templates() -> str:
    lines = ["Classifier templates (labelbench new classifier NAME --template ...):",
             "  rules   keyword rules in a Python file you can extend",
             "  llm     the built-in LLM classifier with prompt templates",
             "",
             "Providers for --template llm (--provider ...):"]
    for p in PROVIDERS:
        lines.append(f"  {p:<18} default model: {DEFAULT_MODELS[p]}")
    lines.append("  openai-compatible works with any server offering an OpenAI-style API;")
    lines.append("  custom creates a Python function to connect any other model.")
    return "\n".join(lines)


# --- building blocks ---------------------------------------------------------

class _TaskInfo:
    """Feature columns and labels of an existing task, used to pre-fill templates."""

    def __init__(self, path: Path) -> None:
        spec_file = path / "task.yaml"
        if not spec_file.is_file():
            raise ScaffoldError(f"{spec_file} not found.")
        spec = read_yaml(spec_file)
        self.name = str(spec.get("name", path.name))
        self.description = str(spec.get("description") or "")
        self.features = list(spec.get("features") or [])
        labels_file = path / spec.get("labels", "labels.csv")
        self.label_columns = csv_columns(labels_file) if labels_file.is_file() else ["label"]
        from labelbench.io import read_csv

        self.labels = [r["label"] for r in read_csv(labels_file)] if labels_file.is_file() else []


class _ExampleTask(_TaskInfo):
    def __init__(self) -> None:
        self.name = "example"
        self.description = "Route customer support messages to the right topic."
        self.features = ["subject", "message"]
        self.label_columns = ["label", "description", "department"]
        self.labels = list(EXAMPLE_KEYWORDS)


def _rules_files(root: Path, name: str, info: _TaskInfo | None,
                 keywords: dict[str, list[str]] | None) -> dict[Path, str]:
    module = name.replace("-", "_")
    if keywords is None:
        labels = info.labels if info and info.labels else ["first_label", "second_label"]
        keywords = {label: [] for label in labels}
    lines = [f"    {label}: [{', '.join(words)}]" for label, words in keywords.items()]
    return {
        root / "classifiers" / f"{module}.py": _template("rules/classifier.py", name=name),
        root / "configs" / f"{name}-v1.yaml": _template("rules/config.yaml", module=module,
                                                        keywords="\n".join(lines)),
    }


def _llm_files(root: Path, name: str, provider: str, model: str | None,
               info: _TaskInfo | None) -> dict[Path, str]:
    if provider not in PROVIDERS:
        raise ScaffoldError(f"Unknown provider {provider!r}; available: {', '.join(PROVIDERS)}.")
    version = f"{name}-v1"
    module = name.replace("-", "_")
    spec = f"classifiers.{module}_provider:complete" if provider == "custom" else provider
    label_line = "- {label}: {description}" if info and "description" in info.label_columns \
        else "- {label}"
    features = info.features if info and info.features else ["text"]
    description = f" {info.description}" if info and info.description else ""
    options = "\n".join("    " + line for line in PROVIDER_OPTIONS[provider])
    files = {
        root / "configs" / f"{version}.yaml": _template(
            "llm/config.yaml", provider=spec, model=model or DEFAULT_MODELS[provider],
            name=version, label_line=label_line, options=options),
        root / "prompts" / f"{version}-system.md": _template(
            "llm/system.md", task=info.name if info else name, description=description),
        root / "prompts" / f"{version}-item.md": "\n".join(
            f"{f.replace('_', ' ').capitalize()}: {{{{{f}}}}}" for f in features) + "\n",
    }
    if provider == "custom":
        files[root / "classifiers" / f"{module}_provider.py"] = _template("llm/custom_provider.py")
    return files


def _template(template_path: str, **values: str) -> str:
    parts = template_path.split("/")
    text = resources.files("labelbench").joinpath("templates", *parts).read_text("utf-8")
    for key, value in values.items():
        text = text.replace(f"@@{key}@@", value)
    leftover = re.findall(r"@@(\w+)@@", text)
    if leftover:
        raise AssertionError(f"Template {template_path} has unfilled placeholders {leftover}.")
    return text


def _check_name(name: str) -> None:
    if not NAME.match(name):
        raise ScaffoldError(f"Invalid name {name!r}: use lower-case letters, digits, '-' or '_', "
                            "starting with a letter.")


def _write(files: dict[Path, str]) -> list[Path]:
    existing = [p for p in files if p.exists()]
    if existing:
        shown = "\n  ".join(str(p) for p in existing)
        raise ScaffoldError(f"Nothing was written, because these files already exist:\n  {shown}")
    for path, text in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
    return list(files)
