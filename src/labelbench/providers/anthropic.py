"""Anthropic Messages API provider."""

from __future__ import annotations

import os
from typing import Any

from labelbench.providers import Completion, ProviderError, require

FALLBACK_BETA = "server-side-fallback-2026-07-01"


class AnthropicProvider:
    """Anthropic Messages API with structured output (output_config.format).

    The system prompt, which holds the label list, is marked for prompt
    caching, so repeated requests pay the full price for it only once.

    Options:
      api_key_env   environment variable with the key; by default the SDK
                    resolves ANTHROPIC_API_KEY or a logged-in `ant` profile
      effort        low | medium | high | xhigh | max
      max_tokens    output limit including thinking, default 4096
      fallbacks     "default" lets the API re-run a request that a model's
                    safety classifier declined on a fallback model; the
                    model that actually answered is recorded per item
      max_retries   SDK retries for rate limits and server errors
    """

    name = "anthropic"

    def __init__(self, model: str, api_key_env: str | None = None, effort: str | None = None,
                 max_tokens: int = 4096, fallbacks: str | None = None,
                 max_retries: int = 4) -> None:
        anthropic = require("anthropic", "anthropic")
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self.fallbacks = fallbacks
        kwargs: dict[str, Any] = {"max_retries": max_retries}
        if api_key_env:
            key = os.environ.get(api_key_env)
            if not key:
                raise ProviderError(f"Environment variable {api_key_env} is not set.")
            kwargs["api_key"] = key
        self.client = anthropic.Anthropic(**kwargs)

    def supports_enum(self, labels: list[str]) -> bool:
        return True

    def complete(self, system: str | None, user: str, schema: dict[str, Any]) -> Completion:
        output_config: dict[str, Any] = {"format": {"type": "json_schema", "schema": schema}}
        if self.effort:
            output_config["effort"] = self.effort
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": user}],
            "output_config": output_config,
        }
        if system:
            kwargs["system"] = [{"type": "text", "text": system,
                                 "cache_control": {"type": "ephemeral"}}]
        if self.fallbacks:
            response = self.client.beta.messages.create(
                betas=[FALLBACK_BETA], fallbacks=self.fallbacks, **kwargs)
        else:
            response = self.client.messages.create(**kwargs)

        if response.stop_reason == "refusal":
            category = getattr(response.stop_details, "category", None) if response.stop_details else None
            raise RuntimeError(f"Model declined the request (refusal, category {category}).")
        if response.stop_reason == "max_tokens":
            raise RuntimeError("Output was cut off at max_tokens; raise options.max_tokens.")
        text = next((b.text for b in response.content if b.type == "text"), "")
        usage = response.usage
        return Completion(
            text=text,
            model=response.model,
            usage={
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "cache_read_input_tokens": getattr(usage, "cache_read_input_tokens", None),
                "cache_creation_input_tokens": getattr(usage, "cache_creation_input_tokens", None),
            },
        )
