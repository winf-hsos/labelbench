"""Reading and writing the CSV, YAML and JSON files labelbench works with."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import yaml

# Text fields may be very long; the default limit of 128 KB is too small.
csv.field_size_limit(min(sys.maxsize, 2**31 - 1))


def read_csv(path: str | Path) -> list[dict[str, str]]:
    """Read a CSV file into a list of rows. A UTF-8 byte order mark is ignored."""
    with Path(path).open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def csv_columns(path: str | Path) -> list[str]:
    with Path(path).open(encoding="utf-8-sig", newline="") as fh:
        return next(csv.reader(fh), [])


def write_csv(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with Path(path).open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_yaml(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def write_yaml(path: str | Path, data: dict[str, Any]) -> None:
    text = yaml.safe_dump(data, allow_unicode=True, sort_keys=False)
    Path(path).write_text(text, encoding="utf-8")


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, data: Any) -> None:
    text = json.dumps(data, indent=2, ensure_ascii=False)
    Path(path).write_text(text + "\n", encoding="utf-8")


def write_text_atomic(path: str | Path, text: str) -> None:
    """Write via a temporary file so an interrupted write never leaves half a file."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
