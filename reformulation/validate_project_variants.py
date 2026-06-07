"""Validate generated project-description reformulations."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from reformulation.common import (  # noqa: E402
    PROJECT_JUDGE_SCHEMA,
    EmbeddingScorer,
    add_common_validation_args,
    average,
    base_view_family,
    call_llm_json,
    ensure_out_dirs,
    lexical_overlap,
    normalize_project_df,
    print_terminal_summary,
    read_csv,
    read_prompt,
    save_raw_response,
    safe_float,
    set_deterministic_seed,
    stable_batch_id,
    write_csv,
    write_json,
)

VIEW_CLASSIFIER_SCHEMA = {
    "name": "view_classifier",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["predicted_view_family", "short_reason"],
        "properties": {
            "predicted_view_family": {
                "type": "string",
                "enum": ["creator_focused", "technical", "skill_focused", "compressed"],
            },
            "short_reason": {"type": "string"},
        },
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_validation_args(parser)
    parser.add_argument("--similarity-threshold", type=float, default=0.70)
    parser.add_argument("--self-bleu-threshold", type=float, default=0.85)
    parser.add_argument("--judge-min-average", type=float, default=4.0)
    parser.add_argument("--judge-min-faithfulness", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_deterministic_seed(args.seed)

    data_root = Path(args.data_root)
    out_root = Path(args.out_root)
    paths = ensure_out_dirs(out_root)
    raw_dir = paths["llm_raw"] / "project_variant_judge"

    original_df = normalize_project_df(read_csv(data_root / "project_descriptions.csv"))
    variants_path = paths["processed"] / "project_variants_raw.csv"
    if not variants_path.exists():
        raise FileNotFoundError(f"Expected generated project variants at {variants_path}")
    variants_df = read_csv(variants_path)

    validated_path = paths["processed"] / "validated_project_variants.csv"
    rejected_path = paths["processed"] / "rejected_project_variants.csv"
    summary_path = paths["summaries"] / "project_variant_validation_summary.json"
    if not args.overwrite:
        existing_outputs = [path for path in (validated_path, rejected_path, summary_path) if path.exists()]
        if existing_outputs:
            joined = ", ".join(str(path) for path in existing_outputs)
            raise FileExistsError(f"Existing output files found: {joined}. Pass --overwrite.")

    reference = original_df.set_index("project_name")["project_description"].to_dict()
    scorer = EmbeddingScorer(args.embedding_model)
    judge_prompt = read_prompt("project_judge_prompt.txt")
    classifier_prompt = read_prompt("view_classifier_prompt.txt")

    rows: list[dict[str, object]] = []
    seen_by_project_view: dict[tuple[str, str], list[str]] = {}

    for _, row in variants_df.iterrows():
        project_name = row["project_name"]
        variant_text = row["project_description"]
        view = row["view"]
        view_family = row["view_family"]
        original_text = reference.get(project_name, "")
        if view == "base":
            rows.append(
                {
                    "project_name": project_name,
                    "original_project_description": original_text,
                    "project_description": variant_text,
                    "view": view,
                    "view_family": view_family,
                    "embedding_similarity": 1.0 if variant_text else 0.0,
                    "self_bleu_or_overlap": 0.0,
                    "faithfulness_score": 5,
                    "coherence_score": 5,
                    "relevance_score": 5,
                    "view_alignment_score": 5,
                    "judge_average_score": 5.0,
                    "unsupported_claims": "",
                    "predicted_view_family": "base",
                    "pass_validation": bool(variant_text),
                    "rejection_reason": "" if variant_text else "empty_base_description",
                }
            )
            continue

        overlap_key = (project_name, view_family)
        previous_texts = seen_by_project_view.get(overlap_key, [])
        overlap_score = max((lexical_overlap(variant_text, previous) for previous in previous_texts), default=0.0)
        seen_by_project_view.setdefault(overlap_key, []).append(variant_text)

        embedding_similarity = scorer.similarity(original_text, variant_text)
        rejection_reasons: list[str] = []
        if embedding_similarity < args.similarity_threshold:
            rejection_reasons.append("semantic_drift")
        if overlap_score > args.self_bleu_threshold:
            rejection_reasons.append("near_duplicate")

        user_prompt = (
            f"Original description: {original_text}\n"
            f"Variant description: {variant_text}\n"
            f"Intended view family: {view_family}\n"
            "Judge the variant against the original and return strict JSON."
        )
        try:
            judgment = call_llm_json(
                model=args.model,
                system_prompt=judge_prompt,
                user_prompt=user_prompt,
                json_schema=PROJECT_JUDGE_SCHEMA,
                seed=args.seed,
                debug=args.debug,
            )
            raw_key = stable_batch_id("projectjudge", [project_name, view, args.model])
            save_raw_response(raw_dir, raw_key, judgment.raw_text, judgment.payload)
        except Exception as exc:  # pragma: no cover - API/runtime dependent
            if args.debug:
                print(f"Judge call failed for {project_name} / {view}: {exc}")
            rows.append(
                {
                    "project_name": project_name,
                    "original_project_description": original_text,
                    "project_description": variant_text,
                    "view": view,
                    "view_family": view_family,
                    "embedding_similarity": embedding_similarity,
                    "self_bleu_or_overlap": overlap_score,
                    "faithfulness_score": 0,
                    "coherence_score": 0,
                    "relevance_score": 0,
                    "view_alignment_score": 0,
                    "judge_average_score": 0.0,
                    "unsupported_claims": "",
                    "predicted_view_family": "",
                    "pass_validation": False,
                    "rejection_reason": "; ".join(rejection_reasons + ["judge_api_failed"]),
                }
            )
            continue

        faithfulness = int(judgment.payload["faithfulness_score"])
        coherence = int(judgment.payload["coherence_score"])
        relevance = int(judgment.payload["relevance_score"])
        view_alignment = int(judgment.payload["view_alignment_score"])
        judge_average = average([faithfulness, coherence, relevance, view_alignment])
        unsupported_claims = judgment.payload.get("unsupported_claims", [])
        predicted_view_family = judgment.payload.get("predicted_view_family", "")

        classifier_user_prompt = (
            f"Original description: {original_text}\n"
            f"Rewritten project description: {variant_text}\n"
            "Predict the dominant view family and return strict JSON."
        )
        try:
            classifier_result = call_llm_json(
                model=args.model,
                system_prompt=classifier_prompt,
                user_prompt=classifier_user_prompt,
                json_schema=VIEW_CLASSIFIER_SCHEMA,
                seed=args.seed,
                debug=args.debug,
            )
            classifier_key = stable_batch_id("viewclassifier", [project_name, view, args.model])
            save_raw_response(raw_dir, classifier_key, classifier_result.raw_text, classifier_result.payload)
            predicted_view_family = classifier_result.payload.get("predicted_view_family", predicted_view_family)
        except Exception as exc:  # pragma: no cover - API/runtime dependent
            if args.debug:
                print(f"Classifier call failed for {project_name} / {view}: {exc}")
            rows.append(
                {
                    "project_name": project_name,
                    "original_project_description": original_text,
                    "project_description": variant_text,
                    "view": view,
                    "view_family": view_family,
                    "embedding_similarity": embedding_similarity,
                    "self_bleu_or_overlap": overlap_score,
                    "faithfulness_score": faithfulness,
                    "coherence_score": coherence,
                    "relevance_score": relevance,
                    "view_alignment_score": view_alignment,
                    "judge_average_score": judge_average,
                    "unsupported_claims": "; ".join(unsupported_claims),
                    "predicted_view_family": "",
                    "pass_validation": False,
                    "rejection_reason": "; ".join(rejection_reasons + ["judge_api_failed"]),
                }
            )
            continue

        if judge_average < args.judge_min_average:
            rejection_reasons.append("low_judge_average")
        if faithfulness < args.judge_min_faithfulness:
            rejection_reasons.append("low_faithfulness")
        if unsupported_claims:
            rejection_reasons.append("unsupported_claims")
        if predicted_view_family != base_view_family(view):
            rejection_reasons.append("view_family_mismatch")

        rows.append(
            {
                "project_name": project_name,
                "original_project_description": original_text,
                "project_description": variant_text,
                "view": view,
                "view_family": view_family,
                "embedding_similarity": embedding_similarity,
                "self_bleu_or_overlap": overlap_score,
                "faithfulness_score": faithfulness,
                "coherence_score": coherence,
                "relevance_score": relevance,
                "view_alignment_score": view_alignment,
                "judge_average_score": judge_average,
                "unsupported_claims": "; ".join(unsupported_claims),
                "predicted_view_family": predicted_view_family,
                "pass_validation": not rejection_reasons,
                "rejection_reason": "; ".join(rejection_reasons),
            }
        )

    results_df = pd.DataFrame(rows)
    validated_df = results_df[results_df["pass_validation"]].reset_index(drop=True)
    rejected_df = results_df[~results_df["pass_validation"]].reset_index(drop=True)

    write_csv(validated_df, validated_path, overwrite=True)
    write_csv(rejected_df, rejected_path, overwrite=True)

    rejection_counts = Counter()
    for value in rejected_df["rejection_reason"].astype(str):
        for reason in [item.strip() for item in value.split(";") if item.strip()]:
            rejection_counts[reason] += 1

    summary = {
        "number_of_variants_checked": int(len(results_df)),
        "number_accepted": int(len(validated_df)),
        "number_rejected": int(len(rejected_df)),
        "rejection_reasons_and_counts": dict(rejection_counts),
        "average_embedding_similarity": average(
            [safe_float(item) for item in results_df["embedding_similarity"].tolist() if pd.notna(item)]
        ),
        "average_judge_scores_by_dimension": {
            "faithfulness": average(results_df["faithfulness_score"].astype(float).tolist()),
            "coherence": average(results_df["coherence_score"].astype(float).tolist()),
            "relevance": average(results_df["relevance_score"].astype(float).tolist()),
            "view_alignment": average(results_df["view_alignment_score"].astype(float).tolist()),
        },
        "accepted_rejected_counts_per_view_family": {
            family: {
                "accepted": int(len(validated_df[validated_df["view_family"] == family])),
                "rejected": int(len(rejected_df[rejected_df["view_family"] == family])),
            }
            for family in sorted(results_df["view_family"].astype(str).unique())
        },
        "output_file_paths": {
            "validated_project_variants": str(validated_path),
            "rejected_project_variants": str(rejected_path),
            "raw_llm_dir": str(raw_dir),
        },
    }
    write_json(summary, summary_path, overwrite=True)
    print_terminal_summary(
        "Project Variant Validation",
        [
            f"checked: {len(results_df)}",
            f"accepted: {len(validated_df)}",
            f"rejected: {len(rejected_df)}",
            f"avg similarity: {summary['average_embedding_similarity']:.4f}",
            f"validated CSV: {validated_path}",
        ],
    )


if __name__ == "__main__":
    main()
