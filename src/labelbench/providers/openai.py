"""OpenAI and OpenAI-compatible providers."""

from __future__ import annotations

import os
from typing import Any

from labelbench.providers import Completion, ProviderError, require

# OpenAI strict schemas allow at most 7,500 characters across all enum values
# once an enum has more than 250 of them.
ENUM_MAX_VALUES = 250
ENUM_MAX_CHARS = 7500


class OpenAIProvider:
    """OpenAI Responses API with a strict JSON schema.

    Options: api_key_env (default OPENAI_API_KEY), reasoning_effort,
    max_output_tokens, max_retries.
    """

    name = "openai"

    def __init__(self, model: str, api_key_env: str = "OPENAI_API_KEY",
                 reasoning_effort: str | None = None, max_output_tokens: int | None = None,
                 max_retries: int = 4) -> None:
        openai = require("openai", "openai")
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.max_output_tokens = max_output_tokens
        self.client = openai.OpenAI(api_key=_key(api_key_env), max_retries=max_retries)

    def supports_enum(self, labels: list[str]) -> bool:
        return len(labels) <= ENUM_MAX_VALUES or sum(map(len, labels)) <= ENUM_MAX_CHARS

    def complete(self, system: str | None, user: str, schema: dict[str, Any]) -> Completion:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "input": user,
            "text": {"format": {"type": "json_schema", "name": "classification",
                                "schema": schema, "strict": True}},
        }
        if system:
            kwargs["instructions"] = system
        if self.reasoning_effort:
            kwargs["reasoning"] = {"effort": self.reasoning_effort}
        if self.max_output_tokens:
            kwargs["max_output_tokens"] = self.max_output_tokens
        response = self.client.responses.create(**kwargs)
        usage = response.usage
        return Completion(
            text=response.output_text,
            model=response.model,
            usage={
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "cached_input_tokens": getattr(usage.input_tokens_details, "cached_tokens", None),
                "reasoning_tokens": getattr(usage.output_tokens_details, "reasoning_tokens", None),
            },
        )


class OpenAICompatibleProvider:
    """Any server with an OpenAI-style /v1/chat/completions endpoint.

    Options: base_url (required), api_key_env (optional; local servers often
    need none), max_tokens, temperature, json_schema (set to false if the
    server does not support structured output; the label is then parsed from
    free JSON text and invalid labels count as errors), max_retries.
    """

    name = "openai-compatible"

    def __init__(self, model: str, base_url: str | None = None, api_key_env: str | None = None,
                 max_tokens: int | None = None, temperature: float | None = None,
                 json_schema: bool = True, max_retries: int = 4) -> None:
        if not base_url:
            raise ProviderError("openai-compatible needs options.base_url, "
                                "e.g. http://localhost:11434/v1")
        openai = require("openai", "openai")
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.json_schema = json_schema
        key = _key(api_key_env) if api_key_env else "not-needed"
        self.client = openai.OpenAI(base_url=base_url, api_key=key, max_retries=max_retries)

    def supports_enum(self, labels: list[str]) -> bool:
        return self.json_schema

    def complete(self, system: str | None, user: str, schema: dict[str, Any]) -> Completion:
        messages = [{"role": "system", "content": system}] if system else []
        messages.append({"role": "user", "content": user})
        kwargs: dict[str, Any] = {"model": self.model, "messages": messages}
        if self.json_schema:
            kwargs["response_format"] = {"type": "json_schema", "json_schema": {
                "name": "classification", "schema": schema, "strict": True}}
        else:
            kwargs["response_format"] = {"type": "json_object"}
        if self.max_tokens:
            kwargs["max_tokens"] = self.max_tokens
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        response = self.client.chat.completions.create(**kwargs)
        usage = response.usage
        return Completion(
            text=response.choices[0].message.content or "",
            model=response.model,
            usage={"input_tokens": getattr(usage, "prompt_tokens", None),
                   "output_tokens": getattr(usage, "completion_tokens", None)} if usage else {},
        )


def _key(env: str) -> str:
    value = os.environ.get(env)
    if not value:
        raise ProviderError(f"Environment variable {env} is not set.")
    return value
