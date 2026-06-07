"""Generate meaning-preserving skill keyword reformulations."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from reformulation.common import (  # noqa: E402
    add_common_generation_args,
    build_skill_generation_schema,
    call_llm_json,
    ensure_out_dirs,
    normalize_skill_df,
    normalize_space,
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
    add_common_generation_args(parser)
    parser.add_argument("--num-variants", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_deterministic_seed(args.seed)

    data_root = Path(args.data_root)
    out_root = Path(args.out_root)
    paths = ensure_out_dirs(out_root)
    raw_dir = paths["llm_raw"] / "skill_variants"

    skill_path = resolve_skill_file(data_root)
    expanded_output_path = paths["processed"] / "expanded_skills.csv"
    combined_output_path = paths["processed"] / "combined_skill_variants.csv"
    summary_path = paths["summaries"] / "skill_variant_generation_summary.json"

    if not args.overwrite and not args.resume:
        existing_outputs = [
            path for path in (expanded_output_path, combined_output_path, summary_path) if path.exists()
        ]
        if existing_outputs:
            joined = ", ".join(str(path) for path in existing_outputs)
            raise FileExistsError(f"Existing output files found: {joined}. Pass --resume or --overwrite.")

    skills_df = normalize_skill_df(read_csv(skill_path))
    existing_expanded = pd.DataFrame()
    if args.resume and expanded_output_path.exists() and not args.overwrite:
        existing_expanded = read_csv(expanded_output_path)

    existing_ids = set(existing_expanded["skill_id"].astype(str)) if not existing_expanded.empty else set()
    expanded_rows: list[dict[str, str]] = []
    combined_rows: list[dict[str, str]] = []
    skipped_skills = 0
    failed_api_calls = 0

    system_prompt = read_prompt("skill_generation_prompt.txt").replace(
        "{{NUM_VARIANTS}}", str(args.num_variants)
    )
    for _, row in skills_df.iterrows():
        skill_id = str(row["id"])
        if args.resume and not args.overwrite and skill_id in existing_ids:
            skipped_skills += 1
            continue
        user_prompt = (
            f"Skill ID: {skill_id}\n"
            f"Canonical skill keyword: {row['keyword']}\n"
            f"Skill level: {row['level']}\n"
            f"Skill categories: {row['categories']}\n"
            f"One-sentence definition: {row['one_sentence_definition']}\n"
            f"Generate exactly {args.num_variants} short keyword-phrase variants."
        )
        try:
            result = call_llm_json(
                model=args.model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                json_schema=build_skill_generation_schema(args.num_variants),
                seed=args.seed,
                debug=args.debug,
            )
            batch_id = stable_batch_id("skillgen", [skill_id, row["keyword"], args.model])
            save_raw_response(raw_dir, batch_id, result.raw_text, result.payload)
            variants = [normalize_space(item) for item in result.payload["reformulated_skill_keywords"]]
            variants = variants[: args.num_variants]
            expanded_rows.append(
                {
                    "skill_id": skill_id,
                    "original_skill_keyword": row["keyword"],
                    "reformulated_skill_keywords": "; ".join(variants),
                }
            )
            for variant in variants:
                combined_rows.append(
                    {
                        "skill_id": skill_id,
                        "base_keyword": row["keyword"],
                        "original_skill_keyword": row["keyword"],
                        "paraphrased_keyword": variant,
                        "source_file": skill_path.name,
                        "model": args.model,
                    }
                )
        except Exception as exc:  # pragma: no cover - API/runtime dependent
            failed_api_calls += 1
            if args.debug:
                print(f"Failed to generate skill variants for {skill_id}: {exc}")

    if args.resume and not existing_expanded.empty and not args.overwrite:
        expanded_df = pd.concat([existing_expanded, pd.DataFrame(expanded_rows)], ignore_index=True)
        expanded_df = expanded_df.drop_duplicates(subset=["skill_id"], keep="last")
    else:
        expanded_df = pd.DataFrame(
            expanded_rows,
            columns=["skill_id", "original_skill_keyword", "reformulated_skill_keywords"],
        )

    if args.resume and combined_output_path.exists() and not args.overwrite:
        existing_combined = read_csv(combined_output_path)
        combined_df = pd.concat([existing_combined, pd.DataFrame(combined_rows)], ignore_index=True)
        combined_df = combined_df.drop_duplicates(subset=["skill_id", "paraphrased_keyword"], keep="last")
    else:
        combined_df = pd.DataFrame(
            combined_rows,
            columns=[
                "skill_id",
                "base_keyword",
                "original_skill_keyword",
                "paraphrased_keyword",
                "source_file",
                "model",
            ],
        )

    expanded_df = expanded_df.sort_values(["skill_id"]).reset_index(drop=True)
    combined_df = combined_df.sort_values(["skill_id", "paraphrased_keyword"]).reset_index(drop=True)

    write_csv(expanded_df, expanded_output_path, overwrite=True)
    write_csv(combined_df, combined_output_path, overwrite=True)

    summary = {
        "number_of_input_skills": int(len(skills_df)),
        "number_of_generated_skill_rows": int(len(expanded_df)),
        "number_of_generated_paraphrases": int(len(combined_df)),
        "average_paraphrases_per_skill": float(len(combined_df) / len(expanded_df)) if len(expanded_df) else 0.0,
        "number_of_failed_api_calls": failed_api_calls,
        "number_of_skipped_skills_due_to_resume": skipped_skills,
        "output_file_paths": {
            "expanded_skills": str(expanded_output_path),
            "combined_skill_variants": str(combined_output_path),
            "raw_llm_dir": str(raw_dir),
        },
    }
    write_json(summary, summary_path, overwrite=True)
    print_terminal_summary(
        "Skill Variant Generation",
        [
            f"input skills: {len(skills_df)}",
            f"expanded rows: {len(expanded_df)}",
            f"paraphrases: {len(combined_df)}",
            f"failed API calls: {failed_api_calls}",
            f"resume skips: {skipped_skills}",
        ],
    )


if __name__ == "__main__":
    main()
