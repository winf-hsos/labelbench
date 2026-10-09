"""Built-in classifier for OpenAI's Decisions API: `classifier: decision`.

The Decisions API answers a `choice` question with the chosen option and a
probability for every option, so this classifier always knows how sure it
is; with `probability: true` in the config the probability of the chosen
label is reported.

A choice question accepts at most 255 options. Longer label lists are split
into several questions of one request, each with an extra option
"none of these". The probability of a label from part i is then

    P(label) ∝ p_i(label) · Π_{j≠i} p_j(none of these)

normalised over all labels: a label is likely if its own part chooses it and
every other part says that none of its labels fits.

Params:
  model            only gpt-6-luna is supported by the API so far
  instructions     file with the question, e.g. "Which standard dish fits best?"
  prompt           file with the item template, {{<feature>}} placeholders
  choice_description  optional format for the description of each option,
                   with fields of labels.csv, e.g. "{description}"
  top_k            number of candidates kept for top-k metrics and the report
  max_choices      options per question, 255 is the API limit
"""

from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from typing import Any

from labelbench.cache import JsonCache
from labelbench.classifier import Prediction, TaskInfo
from labelbench.llm import check_placeholders, render
from labelbench.providers import ProviderError, require

NONE = "__none_of_these__"


class DecisionClassifier:
    def __init__(self, instructions: str, prompt: str, model: str = "gpt-6-luna",
                 choice_description: str | None = None, top_k: int = 5,
                 max_choices: int = 255, none_description: str = "none of these",
                 max_workers: int = 8, api_key_env: str = "OPENAI_API_KEY",
                 max_retries: int = 4, cache_dir: str = ".labelbench-cache") -> None:
        openai = require("openai", "openai")
        key = os.environ.get(api_key_env)
        if not key:
            raise ProviderError(f"Environment variable {api_key_env} is not set.")
        self.client = openai.OpenAI(api_key=key, max_retries=max_retries)
        self.model = model
        self.instructions = Path(instructions).read_text(encoding="utf-8").strip()
        self.item_template = Path(prompt).read_text(encoding="utf-8")
        self.choice_description = choice_description
        self.top_k = top_k
        self.max_choices = max_choices
        self.none_description = none_description
        self.max_workers = max_workers
        self.cache = JsonCache(cache_dir)
        self.probability = False
        self.parts: list[list[str]] = []
        self.questions: list[dict[str, Any]] = []

    def enable_probability(self) -> None:
        self.probability = True

    def prepare(self, task: TaskInfo) -> None:
        check_placeholders({"prompt": self.item_template}, task.features)
        labels = [row["label"] for row in task.labels]
        described = {row["label"]: row for row in task.labels}
        if len(labels) <= self.max_choices:
            self.parts = [labels]
        else:
            # balanced parts, each leaving room for the extra "none of these" option
            size = self.max_choices - 1
            n = -(-len(labels) // size)
            per = -(-len(labels) // n)
            self.parts = [labels[i:i + per] for i in range(0, len(labels), per)]
        split = len(self.parts) > 1
        self.questions = []
        for i, part in enumerate(self.parts):
            choices = []
            for label in part:
                choice: dict[str, Any] = {"value": label}
                if self.choice_description:
                    choice["description"] = self.choice_description.format(**described[label])
                choices.append(choice)
            if split:
                choices.append({"value": NONE, "description": self.none_description})
            self.questions.append({"type": "choice", "name": f"part{i}",
                                   "instructions": self.instructions, "choices": choices})

    def predict(self, items: list[dict[str, str]]) -> list[Prediction]:
        if not self.questions:
            raise RuntimeError("prepare() was not called, so the label list is missing.")
        done, lock = 0, Lock()

        def one(item: dict[str, str]) -> Prediction:
            nonlocal done
            prediction = self._classify(item)
            with lock:
                done += 1
                if done % 25 == 0 or done == len(items):
                    print(f"  {done}/{len(items)} items classified", file=sys.stderr)
            return prediction

        with ThreadPoolExecutor(self.max_workers) as pool:
            return list(pool.map(one, items))

    # --- internals ---------------------------------------------------------

    def _classify(self, item: dict[str, str]) -> Prediction:
        text = render(self.item_template, item)
        key = JsonCache.key(api="decisions", model=self.model, input=text, questions=self.questions)
        cached = self.cache.get(key)
        from_cache = cached is not None
        if cached is None:
            started = time.perf_counter()
            try:
                response = self.client.decisions.create(model=self.model, input=text,
                                                        questions=self.questions)
            except Exception as exc:  # noqa: BLE001 - a failed request counts as an invalid answer
                return Prediction(label=None, raw=f"ERROR: {exc}",
                                  meta={"error": f"{type(exc).__name__}: {exc}"})
            cached = {
                "answers": [_answer(a) for a in response.answers],
                "model": response.model,
                "usage": {"input_tokens": response.usage.input_tokens,
                          "output_tokens": response.usage.output_tokens},
                "latency_s": round(time.perf_counter() - started, 2),
            }
            self.cache.put(key, cached)

        meta = {**cached["usage"], "model": cached["model"], "latency_s": cached["latency_s"],
                "cached": from_cache, "parts": len(self.parts)}
        answers = cached["answers"]
        if any(a["type"] == "refusal" for a in answers):
            return Prediction(label=None, raw=json.dumps({"refusal": True}), meta=meta)

        scores = combine(self.parts, answers)
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        label, p = ranked[0]
        top = ranked[:self.top_k]
        meta["candidate_probabilities"] = {lab: round(q, 4) for lab, q in top}
        meta["probability_source"] = "Decisions API"
        raw = {"label": label, "probability": round(p, 4),
               "parts": [{"name": a["name"], "choice": a["choice"],
                          "confidence": a["confidence"],
                          "none_of_these": a["probabilities"].get(NONE)} for a in answers]}
        return Prediction(label=label, probability=p if self.probability else None,
                          candidates=[lab for lab, _ in top],
                          raw=json.dumps(raw, ensure_ascii=False), meta=meta)


def combine(parts: list[list[str]], answers: list[dict[str, Any]]) -> dict[str, float]:
    """Probability per label from one answer per part (see module docstring)."""
    by_name = {a["name"]: a for a in answers}
    probs = [by_name[f"part{i}"]["probabilities"] for i in range(len(parts))]
    if len(parts) == 1:
        total = sum(probs[0].get(lab, 0.0) for lab in parts[0]) or 1.0
        return {lab: probs[0].get(lab, 0.0) / total for lab in parts[0]}
    scores: dict[str, float] = {}
    for i, part in enumerate(parts):
        others = 1.0
        for j in range(len(parts)):
            if j != i:
                others *= probs[j].get(NONE, 0.0)
        for lab in part:
            scores[lab] = probs[i].get(lab, 0.0) * others
    total = sum(scores.values())
    if total == 0:
        # every part claims one of its labels: fall back to each part's own probabilities
        scores = {lab: probs[i].get(lab, 0.0) for i, part in enumerate(parts) for lab in part}
        total = sum(scores.values()) or 1.0
    return {lab: s / total for lab, s in scores.items()}


def _answer(a: Any) -> dict[str, Any]:
    if a.type != "choice":
        return {"type": a.type, "name": a.name}
    return {"type": "choice", "name": a.name, "choice": a.choice, "confidence": a.confidence,
            "probabilities": {p.value: p.probability for p in a.probabilities}}
