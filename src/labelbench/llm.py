"""The built-in LLM classifier: `classifier: llm` in a config file.

Prompts are two templates. The system template holds everything that is the
same for every item, typically the instructions and the label list; the item
template holds the item's features. Keeping the label list in the system part
lets providers cache it, which makes long label lists affordable.

Placeholders in double braces:
  {{labels}}     the label list, one line per label rendered with `label_line`
  {{<column>}}   a feature column of the item, e.g. {{name}}

The answer is a JSON object with the chosen `label` and, unless `reasoning`
is false, a short `reasoning`. Where the provider allows it, the schema
restricts `label` to the labels in labels.csv, so the model cannot invent one.
"""

from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from typing import Any

from labelbench.cache import JsonCache
from labelbench.classifier import Prediction, TaskInfo
from labelbench.providers import make_provider


class LLMClassifier:
    def __init__(self, provider: str, model: str, prompt: str,
                 system_prompt: str | None = None,
                 label_line: str = "- {label}",
                 reasoning: bool = True,
                 options: dict[str, Any] | None = None,
                 max_workers: int = 8,
                 cache_dir: str = ".labelbench-cache") -> None:
        self.provider_name = provider
        self.model = model
        self.options = dict(options or {})
        self.item_template = Path(prompt).read_text(encoding="utf-8")
        self.system_template = Path(system_prompt).read_text(encoding="utf-8") if system_prompt else None
        self.label_line = label_line
        self.reasoning = reasoning
        self.max_workers = max_workers
        self.cache = JsonCache(cache_dir)
        self.provider = make_provider(provider, model, self.options)
        self.labels: list[str] = []
        self.label_text = ""

    def prepare(self, task: TaskInfo) -> None:
        self.labels = [row["label"] for row in task.labels]
        try:
            self.label_text = "\n".join(self.label_line.format(**row) for row in task.labels)
        except KeyError as exc:
            raise ValueError(f"label_line uses {exc}, which is not a column of labels.csv.") from exc
        self._check_placeholders(task.features)

    def predict(self, items: list[dict[str, str]]) -> list[Prediction]:
        if not self.labels:
            raise RuntimeError("prepare() was not called, so the label list is missing.")
        schema = self._schema()
        done, lock = 0, Lock()

        def one(item: dict[str, str]) -> Prediction:
            nonlocal done
            prediction = self._classify(item, schema)
            with lock:
                done += 1
                if done % 10 == 0 or done == len(items):
                    print(f"  {done}/{len(items)} items classified", file=sys.stderr)
            return prediction

        with ThreadPoolExecutor(self.max_workers) as pool:
            return list(pool.map(one, items))

    # --- internals ---------------------------------------------------------

    def _render(self, template: str, item: dict[str, str]) -> str:
        text = template.replace("{{labels}}", self.label_text)
        for key, value in item.items():
            text = text.replace("{{" + key + "}}", (value or "").strip() or "-")
        return text

    def _check_placeholders(self, features: list[str]) -> None:
        import re

        known = {"labels", "id", *features}
        for name, template in (("prompt", self.item_template), ("system_prompt", self.system_template)):
            if template is None:
                continue
            unknown = sorted(set(re.findall(r"\{\{([^{}]+)\}\}", template)) - known)
            if unknown:
                raise ValueError(f"{name} uses placeholders {unknown} that are neither "
                                 f"'labels' nor feature columns {features}.")

    def _schema(self) -> dict[str, Any]:
        label: dict[str, Any] = {"type": "string",
                                 "description": "Exactly one label from the list, spelled as listed."}
        if self.provider.supports_enum(self.labels):
            label["enum"] = self.labels
        properties: dict[str, Any] = {}
        if self.reasoning:
            properties["reasoning"] = {"type": "string",
                                       "description": "One or two sentences explaining the choice."}
        properties["label"] = label
        return {"type": "object", "properties": properties,
                "required": list(properties), "additionalProperties": False}

    def _classify(self, item: dict[str, str], schema: dict[str, Any]) -> Prediction:
        system = self._render(self.system_template, item) if self.system_template else None
        user = self._render(self.item_template, item)
        key = JsonCache.key(provider=self.provider_name, model=self.model, options=self.options,
                            system=system, user=user, schema=schema)
        cached = self.cache.get(key)
        from_cache = cached is not None
        if cached is None:
            started = time.perf_counter()
            try:
                completion = self.provider.complete(system, user, schema)
            except Exception as exc:  # noqa: BLE001 - a failed request counts as an invalid answer
                return Prediction(label=None, raw=f"ERROR: {exc}",
                                  meta={"error": f"{type(exc).__name__}: {exc}"})
            cached = {**completion, "latency_s": round(time.perf_counter() - started, 2)}
            self.cache.put(key, cached)

        meta = {**(cached.get("usage") or {}), "model": cached.get("model"),
                "latency_s": cached.get("latency_s"), "cached": from_cache,
                "enum": "enum" in schema["properties"]["label"]}
        try:
            answer = json.loads(cached["text"])
            label = answer["label"]
        except (json.JSONDecodeError, KeyError, TypeError):
            return Prediction(label=None, raw=cached.get("text", ""), meta=meta)
        return Prediction(label=label if isinstance(label, str) else None,
                          raw=cached["text"], meta=meta)
