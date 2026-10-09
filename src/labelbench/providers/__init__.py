"""LLM providers for the built-in LLM classifier.

A provider sends one request and returns the model's text. Everything else
(prompt rendering, schema, caching, parallelism, parsing) lives in
labelbench.llm, so adding a provider means implementing one method.

Built-in providers:

- openai              OpenAI Responses API (pip install labelbench[openai])
- anthropic           Anthropic Messages API (pip install labelbench[anthropic])
- openai-compatible   any server with an OpenAI-style chat completions endpoint,
                      e.g. a local model server, selected with base_url
- module:function     your own function for any other provider
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any, Protocol

PROVIDERS = {
    "openai": "labelbench.providers.openai:OpenAIProvider",
    "openai-compatible": "labelbench.providers.openai:OpenAICompatibleProvider",
    "anthropic": "labelbench.providers.anthropic:AnthropicProvider",
}


class Completion(dict):
    """{"text": str, "usage": dict, "model": str} - the model's raw JSON text plus metadata."""


class Provider(Protocol):
    name: str

    def complete(self, system: str | None, user: str, schema: dict[str, Any]) -> Completion: ...

    def supports_enum(self, labels: list[str]) -> bool: ...


class ProviderError(RuntimeError):
    """The provider is unknown, not installed or misconfigured."""


def make_provider(spec: str, model: str, options: dict[str, Any]) -> Provider:
    target = PROVIDERS.get(spec, spec)
    if ":" not in target:
        known = ", ".join(PROVIDERS)
        raise ProviderError(f"Unknown provider {spec!r}. Use one of {known} or module:function.")
    module_name, attr = target.split(":", 1)
    cwd = str(Path.cwd())
    if cwd not in sys.path:
        sys.path.insert(0, cwd)
    module = importlib.import_module(module_name)
    obj = getattr(module, attr, None)
    if obj is None:
        raise ProviderError(f"{module_name} has no attribute {attr!r}.")
    if spec in PROVIDERS:
        return obj(model=model, **options)
    return CustomProvider(spec, obj, model, options)


class CustomProvider:
    """Wraps a plain function so any provider can be used.

    The function receives `system`, `user`, `schema`, `model` and every entry
    of `options` as keyword arguments and returns either the model's text or a
    dict with at least "text" (optionally "usage" and "model").
    """

    def __init__(self, name: str, fn: Any, model: str, options: dict[str, Any]) -> None:
        self.name = name
        self.fn = fn
        self.model = model
        self.options = options

    def complete(self, system: str | None, user: str, schema: dict[str, Any]) -> Completion:
        result = self.fn(system=system, user=user, schema=schema, model=self.model, **self.options)
        if isinstance(result, str):
            return Completion(text=result, usage={}, model=self.model)
        return Completion(text=result["text"], usage=result.get("usage", {}),
                          model=result.get("model", self.model))

    def supports_enum(self, labels: list[str]) -> bool:
        return True


def require(package: str, extra: str) -> Any:
    try:
        return importlib.import_module(package)
    except ImportError as exc:
        raise ProviderError(f"The {package!r} package is missing. Install it with: "
                            f"pip install labelbench[{extra}]") from exc
