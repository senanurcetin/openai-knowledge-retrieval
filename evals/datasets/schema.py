from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from pydantic import BaseModel, Field, ValidationError, field_validator
import json


class EvalItem(BaseModel):
    id: Optional[str] = None
    question: str
    citation_text: str
    correct_answer: str
    source_id: Optional[str] = None
    page: Optional[int] = None
    char_start: Optional[int] = None
    char_end: Optional[int] = None
    difficulty: Optional[str] = None
    tags: List[str] = Field(default_factory=list)

    @field_validator("question", "correct_answer")
    @classmethod
    def require_nonempty_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Evaluation question and correct_answer must be nonempty")
        return value

    @field_validator("id")
    @classmethod
    def require_nonempty_identifier(cls, value: Optional[str]) -> Optional[str]:
        if value is not None and not value.strip():
            raise ValueError("Provided evaluation id must be nonempty")
        return value


class EvalRow(BaseModel):
    item: EvalItem


def load_and_validate_jsonl(path: str | Path) -> List[EvalRow]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found: {p}")
    rows: List[EvalRow] = []
    seen_ids: Dict[str, int] = {}
    with p.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except Exception as e:
                raise ValueError(f"Invalid JSON on line {i}: {e}")

            if not isinstance(obj, dict):
                raise ValueError(
                    f"Invalid JSON object on line {i}: expected dict, got {type(obj)}"
                )

            # Accept both explicit {"item": {...}} and legacy flat records.
            # Never silently ignore recognized flat fields beside "item".
            if "item" in obj:
                flat_keys = set(obj) & set(EvalItem.model_fields)
                if flat_keys:
                    raise ValueError(
                        f"Mixed flat and wrapped evaluation fields on line {i}"
                    )
            else:
                obj = {"item": obj}

            try:
                row = EvalRow(**obj)
            except ValidationError as e:
                raise ValueError(f"Schema error on line {i}: {e}") from e

            record_id = row.item.id
            if record_id is not None:
                first_line = seen_ids.get(record_id)
                if first_line is not None:
                    raise ValueError(
                        f"Duplicate evaluation id {record_id!r} on line {i}; "
                        f"first seen on line {first_line}"
                    )
                seen_ids[record_id] = i
            rows.append(row)
    if not rows:
        raise ValueError("Dataset is empty after parsing")
    return rows


def write_jsonl(rows: Iterable[EvalRow | Dict[str, Any]], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            if isinstance(r, EvalRow):
                data = r.model_dump()
            else:
                data = r
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
