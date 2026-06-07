"""Validate generated skill-keyword reformulations."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from reformulation.common import (  # noqa: E402
    SKILL_JUDGE_SCHEMA,
    EmbeddingScorer,
    add_common_validation_args,
    average,
    call_llm_json,
    ensure_out_dirs,
    flatten_semicolon_list,
    lexical_overlap,
    normalize_skill_df,
    print_terminal_summary,
    read_csv,
    read_prompt,
    resolve_skill_file,
    save_raw_response,
    set_deterministic_seed,
    stable_batch_id,
    write_csv,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_validation_args(parser)
    parser.add_argument("--similarity-threshold", type=float, default=0.50)
    parser.add_argument("--judge-min-average", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_deterministic_seed(args.seed)

    data_root = Path(args.data_root)
    out_root = Path(args.out_root)
    paths = ensure_out_dirs(out_root)
    raw_dir = paths["llm_raw"] / "skill_variant_judge"

    skill_path = resolve_skill_file(data_root)
    skills_df = normalize_skill_df(read_csv(skill_path))
    expanded_path = paths["processed"] / "expanded_skills.csv"
    if not expanded_path.exists():
        raise FileNotFoundError(f"Expected generated skill variants at {expanded_path}")
    expanded_df = read_csv(expanded_path)

    validated_path = paths["processed"] / "validated_skill_variants.csv"
    rejected_path = paths["processed"] / "rejected_skill_variants.csv"
    summary_path = paths["summaries"] / "skill_variant_validation_summary.json"
    if not args.overwrite:
        existing_outputs = [path for path in (validated_path, rejected_path, summary_path) if path.exists()]
        if existing_outputs:
            joined = ", ".join(str(path) for path in existing_outputs)
            raise FileExistsError(f"Existing output files found: {joined}. Pass --overwrite.")

    scorer = EmbeddingScorer(args.embedding_model)
    judge_prompt = read_prompt("skill_judge_prompt.txt")
    reference = skills_df.set_index("id").to_dict(orient="index")

    rows: list[dict[str, object]] = []
    seen_by_skill: dict[str, list[str]] = {}

    for _, row in expanded_df.iterrows():
        skill_id = str(row["skill_id"])
        if skill_id not in reference:
            rows.append(
                {
                    "skill_id": skill_id,
                    "base_keyword": "",
                    "paraphrased_keyword": "",
                    "meaning_preservation_score": 0,
                    "specificity_score": 0,
                    "lexical_quality_score": 0,
                    "non_expansion_score": 0,
                    "judge_average_score": 0.0,
                    "embedding_similarity": 0.0,
                    "pass_validation": False,
                    "rejection_reason": "unknown_skill_id",
                    "skill_category": "",
                }
            )
            continue

        ref = reference[skill_id]
        for paraphrase in flatten_semicolon_list(str(row["reformulated_skill_keywords"])):
            reasons: list[str] = []
            tokens = paraphrase.split()
            if len(tokens) < 2 or len(tokens) > 6:
                reasons.append("short_phrase_check_failed")

            previous = seen_by_skill.get(skill_id, [])
            if paraphrase.lower() == str(ref["keyword"]).lower():
                reasons.append("duplicate_of_base_keyword")
            if max((lexical_overlap(paraphrase, item) for item in previous), default=0.0) > 0.9:
                reasons.append("duplicate_or_near_duplicate")
            seen_by_skill.setdefault(skill_id, []).append(paraphrase)

            compare_text = f"{ref['keyword']} :: {ref['one_sentence_definition']}"
            similarity = scorer.similarity(compare_text, paraphrase)

            user_prompt = (
                f"Original skill keyword: {ref['keyword']}\n"
                f"Skill categories: {ref['categories']}\n"
                f"One-sentence definition: {ref['one_sentence_definition']}\n"
                f"Paraphrased keyword: {paraphrase}\n"
                "Judge whether the paraphrase preserves the same skill meaning and return strict JSON."
            )
            try:
                judgment = call_llm_json(
                    model=args.model,
                    system_prompt=judge_prompt,
                    user_prompt=user_prompt,
                    json_schema=SKILL_JUDGE_SCHEMA,
                    seed=args.seed,
                    debug=args.debug,
                )
                raw_key = stable_batch_id("skilljudge", [skill_id, paraphrase, args.model])
                save_raw_response(raw_dir, raw_key, judgment.raw_text, judgment.payload)
            except Exception as exc:  # pragma: no cover - API/runtime dependent
                if args.debug:
                    print(f"Judge call failed for skill {skill_id} / {paraphrase}: {exc}")
                rows.append(
                    {
                        "skill_id": skill_id,
                        "base_keyword": ref["keyword"],
                        "paraphrased_keyword": paraphrase,
                        "meaning_preservation_score": 0,
                        "specificity_score": 0,
                        "lexical_quality_score": 0,
                        "non_expansion_score": 0,
                        "judge_average_score": 0.0,
                        "embedding_similarity": similarity,
                        "pass_validation": False,
                        "rejection_reason": "; ".join(reasons + ["judge_api_failed"]),
                        "skill_category": ref["categories"],
                    }
                )
                continue

            scores = [
                int(judgment.payload["meaning_preservation_score"]),
                int(judgment.payload["specificity_score"]),
                int(judgment.payload["lexical_quality_score"]),
                int(judgment.payload["non_expansion_score"]),
            ]
            judge_average = average(scores)
            if similarity < args.similarity_threshold and judge_average < args.judge_min_average:
                reasons.append("embedding_drift")
            if judge_average < args.judge_min_average:
                reasons.append("low_judge_average")
            if judgment.payload["is_new_skill"]:
                reasons.append("new_skill")
            if judgment.payload["is_too_broad"]:
                reasons.append("too_broad")
            if judgment.payload["is_too_narrow"]:
                reasons.append("too_narrow")
            if judgment.payload["is_project_specific"]:
                reasons.append("project_specific")
            if judgment.payload["is_role_or_contributor_term"]:
                reasons.append("role_or_contributor_term")

            rows.append(
                {
                    "skill_id": skill_id,
                    "base_keyword": ref["keyword"],
                    "paraphrased_keyword": paraphrase,
                    "meaning_preservation_score": scores[0],
                    "specificity_score": scores[1],
                    "lexical_quality_score": scores[2],
                    "non_expansion_score": scores[3],
                    "judge_average_score": judge_average,
                    "embedding_similarity": similarity,
                    "pass_validation": not reasons,
                    "rejection_reason": "; ".join(reasons),
                    "skill_category": ref["categories"],
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

    category_counts: dict[str, dict[str, int]] = {}
    for category in sorted(results_df["skill_category"].astype(str).unique()):
        category_counts[category] = {
            "accepted": int(len(validated_df[validated_df["skill_category"] == category])),
            "rejected": int(len(rejected_df[rejected_df["skill_category"] == category])),
        }

    summary = {
        "number_of_paraphrases_checked": int(len(results_df)),
        "number_accepted": int(len(validated_df)),
        "number_rejected": int(len(rejected_df)),
        "rejection_reasons_and_counts": dict(rejection_counts),
        "average_embedding_similarity": average(results_df["embedding_similarity"].astype(float).tolist()),
        "average_judge_scores_by_dimension": {
            "meaning_preservation": average(results_df["meaning_preservation_score"].astype(float).tolist()),
            "specificity": average(results_df["specificity_score"].astype(float).tolist()),
            "lexical_quality": average(results_df["lexical_quality_score"].astype(float).tolist()),
            "non_expansion": average(results_df["non_expansion_score"].astype(float).tolist()),
        },
        "accepted_rejected_counts_per_skill_category": category_counts,
        "output_file_paths": {
            "validated_skill_variants": str(validated_path),
            "rejected_skill_variants": str(rejected_path),
            "raw_llm_dir": str(raw_dir),
        },
    }
    write_json(summary, summary_path, overwrite=True)
    print_terminal_summary(
        "Skill Variant Validation",
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
