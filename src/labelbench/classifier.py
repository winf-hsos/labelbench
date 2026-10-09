"""The classifier interface and loading a classifier from its config file."""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from labelbench.io import read_yaml


@dataclass
class Prediction:
    label: str | None
    """The predicted label, or None if the classifier has no valid answer."""
    candidates: list[str] | None = None
    """Further labels in ranked order, used for top-k metrics."""
    probability: float | None = None
    """Probability (0 to 1) that `label` is correct. Only reported if the
    config sets `probability: true`; see enable_probability() below."""
    raw: str = ""
    """Raw output, e.g. the LLM response including its reasoning."""
    meta: dict[str, Any] = field(default_factory=dict)
    """Anything else worth keeping, such as tokens or latency."""


@dataclass(frozen=True)
class TaskInfo:
    """What a classifier may know about the task. Gold labels are deliberately absent."""

    name: str
    type: str
    features: list[str]
    labels: list[dict[str, str]]
    levels: list[str]


class Classifier(Protocol):
    def predict(self, items: list[dict[str, str]]) -> list[Prediction]: ...

    # Optional:
    #   prepare(task: TaskInfo)      called once before predict()
    #   enable_probability()         called if the config sets probability: true;
    #                                a classifier without it cannot deliver probabilities


# Short names for classifiers shipped with labelbench, e.g. {"llm": "labelbench.llm:LLM"}.
BUILTINS: dict[str, str] = {
    "llm": "labelbench.llm:LLMClassifier",
    "decision": "labelbench.decision:DecisionClassifier",
}


class ClassifierError(ValueError):
    """The classifier config cannot be resolved."""


def load_classifier(config_path: str | Path) -> tuple[Classifier, dict[str, Any]]:
    """Instantiate the classifier described in a config file.

    `classifier` is a built-in short name or `module:Class`. Modules are
    imported relative to the current working directory, so project-specific
    classifiers stay in the project that uses labelbench. Everything under
    `params` is passed to the constructor.
    """
    config_path = Path(config_path)
    if not config_path.is_file():
        raise ClassifierError(f"{config_path} not found.")
    config = read_yaml(config_path)
    spec = config.get("classifier")
    if not spec:
        raise ClassifierError(f"{config_path} has no 'classifier' entry.")

    target = BUILTINS.get(spec, spec)
    if ":" not in target:
        raise ClassifierError(f"Unknown classifier {spec!r}: use a built-in name or module:Class.")
    module_name, class_name = target.split(":", 1)

    cwd = str(Path.cwd())
    if cwd not in sys.path:
        sys.path.insert(0, cwd)
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise ClassifierError(f"Cannot import {module_name!r} from {cwd}: {exc}") from exc
    cls = getattr(module, class_name, None)
    if cls is None:
        raise ClassifierError(f"{module_name} has no class {class_name!r}.")

    params = config.get("params") or {}
    return cls(**params), config
