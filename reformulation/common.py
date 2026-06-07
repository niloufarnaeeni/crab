"""Shared utilities for the reformulation and validation pipeline."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import sys
import uuid
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

try:
    import numpy as np
except ImportError:  # pragma: no cover - optional dependency
    np = None

try:
    import pandas as pd
except ImportError as exc:  # pragma: no cover - required dependency
    raise RuntimeError("The pipeline requires pandas to be installed.") from exc


VIEW_FAMILIES = ("creator_focused", "technical", "skill_focused", "compressed")
SKILL_FILE_CANDIDATES = ("gpt5_skills.csv", "skill_gpt5.csv")


@dataclass
class JudgeResult:
    payload: dict[str, Any]
    raw_text: str


def add_common_generation_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--model", default="gpt-5")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--debug", action="store_true")


def add_common_validation_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--model", default="gpt-5")
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--debug", action="store_true")


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def ensure_out_dirs(out_root: Path) -> dict[str, Path]:
    processed = out_root / "processed"
    summaries = out_root / "summaries"
    llm_raw = processed / "llm_raw"
    for directory in (processed, summaries, llm_raw):
        directory.mkdir(parents=True, exist_ok=True)
    return {"processed": processed, "summaries": summaries, "llm_raw": llm_raw}


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, keep_default_na=False)


def write_csv(df: pd.DataFrame, path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists. Pass --overwrite to replace it.")
    ensure_parent(path)
    df.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)


def write_json(data: dict[str, Any], path: Path, overwrite: bool = True) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists. Pass --overwrite to replace it.")
    ensure_parent(path)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def print_terminal_summary(title: str, lines: Iterable[str]) -> None:
    print(f"\n[{title}]")
    for line in lines:
        print(f"- {line}")


def normalize_space(text: Any) -> str:
    value = "" if text is None else str(text)
    return re.sub(r"\s+", " ", value).strip()


def normalize_project_df(df: pd.DataFrame) -> pd.DataFrame:
    required = {"project_name", "project_description"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required project columns: {sorted(missing)}")
    normalized = df.copy()
    normalized["project_name"] = normalized["project_name"].map(normalize_space)
    normalized["project_description"] = normalized["project_description"].map(normalize_space)
    normalized = normalized[normalized["project_name"] != ""].reset_index(drop=True)
    return normalized


def resolve_skill_file(data_root: Path) -> Path:
    for name in SKILL_FILE_CANDIDATES:
        candidate = data_root / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find any base skill file in {data_root}. Expected one of {SKILL_FILE_CANDIDATES}."
    )


def normalize_skill_df(df: pd.DataFrame) -> pd.DataFrame:
    required = {"id", "keyword", "level", "categories", "one_sentence_definition"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required skill columns: {sorted(missing)}")
    normalized = df.copy()
    for column in required:
        normalized[column] = normalized[column].map(normalize_space)
    normalized = normalized[normalized["id"] != ""].reset_index(drop=True)
    return normalized


def set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    if np is not None:
        np.random.seed(seed)


def stable_batch_id(prefix: str, values: Iterable[str]) -> str:
    base = "|".join(values)
    return f"{prefix}_{uuid.uuid5(uuid.NAMESPACE_DNS, base).hex[:12]}"


def read_prompt(prompt_name: str) -> str:
    prompt_path = Path(__file__).resolve().parent / "prompts" / prompt_name
    return prompt_path.read_text(encoding="utf-8")


def maybe_extract_json(text: str) -> dict[str, Any]:
    stripped = text.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def load_openai_client():
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise RuntimeError("The pipeline requires the openai package to call GPT models.") from exc
    return OpenAI()


def call_llm_json(
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    json_schema: dict[str, Any] | None,
    seed: int,
    debug: bool,
) -> JudgeResult:
    client = load_openai_client()
    raw_text = ""
    payload: dict[str, Any]
    if hasattr(client, "responses"):
        response_kwargs: dict[str, Any] = {
            "model": model,
            "input": [
                {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
                {"role": "user", "content": [{"type": "input_text", "text": user_prompt}]},
            ],
            "seed": seed,
        }
        if json_schema:
            response_kwargs["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": json_schema.get("name", "response"),
                    "schema": json_schema["schema"],
                    "strict": True,
                }
            }
        response = client.responses.create(**response_kwargs)
        raw_text = getattr(response, "output_text", "") or ""
        if not raw_text:
            raw_text = json.dumps(response.model_dump(), indent=2)
        payload = maybe_extract_json(raw_text)
    else:  # pragma: no cover - fallback
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": json_schema.get("name", "response") if json_schema else "response",
                    "schema": (json_schema or {"schema": {"type": "object"}})["schema"],
                    "strict": True,
                },
            }
            if json_schema
            else {"type": "json_object"},
            seed=seed,
        )
        raw_text = response.choices[0].message.content or "{}"
        payload = maybe_extract_json(raw_text)
    if debug:
        print("LLM payload:", json.dumps(payload, indent=2))
    return JudgeResult(payload=payload, raw_text=raw_text)


def save_raw_response(raw_dir: Path, key: str, raw_text: str, payload: dict[str, Any]) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    safe_key = re.sub(r"[^A-Za-z0-9_.-]+", "_", key)
    path = raw_dir / f"{safe_key}.json"
    path.write_text(
        json.dumps({"raw_text": raw_text, "parsed_payload": payload}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return path


def tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+", text.lower())


def lexical_overlap(text_a: str, text_b: str) -> float:
    tokens_a = set(tokenize(text_a))
    tokens_b = set(tokenize(text_b))
    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


def average(values: Iterable[float]) -> float:
    numbers = list(values)
    if not numbers:
        return 0.0
    return sum(numbers) / len(numbers)


class EmbeddingScorer:
    """Compute cosine similarity with sentence-transformers when available."""

    def __init__(self, model_name: str) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - environment-specific
            raise RuntimeError(
                "Validation requires sentence-transformers to be installed."
            ) from exc
        self.model = SentenceTransformer(model_name)

    def similarity(self, left: str, right: str) -> float:
        embeddings = self.model.encode([left, right], normalize_embeddings=True)
        left_vec, right_vec = embeddings[0], embeddings[1]
        if np is not None:
            return float(np.dot(left_vec, right_vec))
        return sum(float(a) * float(b) for a, b in zip(left_vec, right_vec))


def summarize_counts(values: Iterable[str]) -> dict[str, int]:
    return dict(Counter(values))


def flatten_semicolon_list(value: str) -> list[str]:
    return [normalize_space(part) for part in value.split(";") if normalize_space(part)]


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def base_view_family(view: str) -> str:
    if view == "base":
        return "base"
    return re.sub(r"_\d+$", "", view)


def build_project_generation_schema(num_variants_per_view: int) -> dict[str, Any]:
    properties = {
        family: {
            "type": "array",
            "minItems": num_variants_per_view,
            "maxItems": num_variants_per_view,
            "items": {"type": "string"},
        }
        for family in VIEW_FAMILIES
    }
    return {
        "name": "project_variants",
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": list(VIEW_FAMILIES),
            "properties": properties,
        },
    }


def build_skill_generation_schema(num_variants: int) -> dict[str, Any]:
    return {
        "name": "skill_variants",
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["reformulated_skill_keywords"],
            "properties": {
                "reformulated_skill_keywords": {
                    "type": "array",
                    "minItems": num_variants,
                    "maxItems": num_variants,
                    "items": {"type": "string"},
                }
            },
        },
    }


PROJECT_JUDGE_SCHEMA = {
    "name": "project_variant_judgment",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "faithfulness_score",
            "coherence_score",
            "relevance_score",
            "view_alignment_score",
            "unsupported_claims",
            "predicted_view_family",
            "short_reason",
        ],
        "properties": {
            "faithfulness_score": {"type": "integer", "minimum": 1, "maximum": 5},
            "coherence_score": {"type": "integer", "minimum": 1, "maximum": 5},
            "relevance_score": {"type": "integer", "minimum": 1, "maximum": 5},
            "view_alignment_score": {"type": "integer", "minimum": 1, "maximum": 5},
            "unsupported_claims": {"type": "array", "items": {"type": "string"}},
            "predicted_view_family": {
                "type": "string",
                "enum": list(VIEW_FAMILIES),
            },
            "short_reason": {"type": "string"},
        },
    },
}


SKILL_JUDGE_SCHEMA = {
    "name": "skill_variant_judgment",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "meaning_preservation_score",
            "specificity_score",
            "lexical_quality_score",
            "non_expansion_score",
            "is_new_skill",
            "is_too_broad",
            "is_too_narrow",
            "is_project_specific",
            "is_role_or_contributor_term",
            "short_reason",
        ],
        "properties": {
            "meaning_preservation_score": {"type": "integer", "minimum": 1, "maximum": 5},
            "specificity_score": {"type": "integer", "minimum": 1, "maximum": 5},
            "lexical_quality_score": {"type": "integer", "minimum": 1, "maximum": 5},
            "non_expansion_score": {"type": "integer", "minimum": 1, "maximum": 5},
            "is_new_skill": {"type": "boolean"},
            "is_too_broad": {"type": "boolean"},
            "is_too_narrow": {"type": "boolean"},
            "is_project_specific": {"type": "boolean"},
            "is_role_or_contributor_term": {"type": "boolean"},
            "short_reason": {"type": "string"},
        },
    },
}
