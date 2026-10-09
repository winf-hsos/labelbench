"""A small disk cache for expensive classifier calls such as LLM requests.

The key is a hash over everything that determines the answer (model, prompt,
parameters, input), so a re-run only pays for what actually changed. Each
entry is one JSON file, written atomically, which makes the cache safe to use
from several threads.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from labelbench.io import sha256_text, write_text_atomic


class JsonCache:
    def __init__(self, directory: str | Path = ".labelbench-cache") -> None:
        self.directory = Path(directory)

    @staticmethod
    def key(**parts: Any) -> str:
        return sha256_text(json.dumps(parts, sort_keys=True, ensure_ascii=False))

    def _path(self, key: str) -> Path:
        return self.directory / key[:2] / f"{key}.json"

    def get(self, key: str) -> Any | None:
        path = self._path(key)
        if not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def put(self, key: str, value: Any) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_text_atomic(path, json.dumps(value, ensure_ascii=False))
