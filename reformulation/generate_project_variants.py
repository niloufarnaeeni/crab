"""Generate project-description reformulations for multiple view families."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from reformulation.common import (  # noqa: E402
    VIEW_FAMILIES,
    add_common_generation_args,
    build_project_generation_schema,
    call_llm_json,
    ensure_out_dirs,
    normalize_project_df,
    normalize_space,
    print_terminal_summary,
    read_csv,
    read_prompt,
    save_raw_response,
    set_deterministic_seed,
    stable_batch_id,
    write_csv,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_generation_args(parser)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--views", nargs="+", default=list(VIEW_FAMILIES))
    parser.add_argument("--num-variants-per-view", type=int, default=3)
    return parser.parse_args()


def build_user_prompt(project_name: str, project_description: str, num_variants_per_view: int) -> str:
    return (
        f"Project name: {project_name}\n"
        f"Original project description: {project_description}\n"
        f"Generate exactly {num_variants_per_view} meaning-preserving variants for each required view family."
    )


def generate_variants_for_row(
    *,
    row: pd.Series,
    args: argparse.Namespace,
    raw_dir: Path,
) -> tuple[list[dict[str, str]], bool]:
    system_prompt = read_prompt("project_generation_prompt.txt").replace(
        "{{NUM_VARIANTS_PER_VIEW}}", str(args.num_variants_per_view)
    )
    result = call_llm_json(
        model=args.model,
        system_prompt=system_prompt,
        user_prompt=build_user_prompt(
            row["project_name"], row["project_description"], args.num_variants_per_view
        ),
        json_schema=build_project_generation_schema(args.num_variants_per_view),
        seed=args.seed,
        debug=args.debug,
    )
    batch_id = stable_batch_id("projectgen", [row["project_name"], row["project_description"], args.model])
    save_raw_response(raw_dir, batch_id, result.raw_text, result.payload)
    records: list[dict[str, str]] = []
    for family in args.views:
        variants = [normalize_space(item) for item in result.payload.get(family, [])]
        for index, variant in enumerate(variants, start=1):
            records.append(
                {
                    "project_name": row["project_name"],
                    "project_description": variant,
                    "view": f"{family}_{index}",
                    "view_family": family,
                    "source": "gpt-5-reformulation",
                    "model": args.model,
                    "generation_batch_id": batch_id,
                }
            )
    return records, True


def main() -> None:
    args = parse_args()
    set_deterministic_seed(args.seed)

    data_root = Path(args.data_root)
    out_root = Path(args.out_root)
    paths = ensure_out_dirs(out_root)
    raw_dir = paths["llm_raw"] / "project_variants"

    input_path = data_root / "project_descriptions.csv"
    raw_output_path = paths["processed"] / "project_variants_raw.csv"
    by_view_output_path = paths["processed"] / "project_variants_by_view.csv"
    summary_path = paths["summaries"] / "project_variant_generation_summary.json"

    if not args.overwrite and not args.resume:
        existing_outputs = [path for path in (raw_output_path, by_view_output_path, summary_path) if path.exists()]
        if existing_outputs:
            joined = ", ".join(str(path) for path in existing_outputs)
            raise FileExistsError(f"Existing output files found: {joined}. Pass --resume or --overwrite.")

    existing_df = pd.DataFrame()
    if args.resume and raw_output_path.exists() and not args.overwrite:
        existing_df = read_csv(raw_output_path)

    projects_df = normalize_project_df(read_csv(input_path))
    base_rows = [
        {
            "project_name": row["project_name"],
            "project_description": row["project_description"],
            "view": "base",
            "view_family": "base",
            "source": "base",
            "model": "",
            "generation_batch_id": "",
        }
        for _, row in projects_df.iterrows()
    ]

    generated_rows: list[dict[str, str]] = []
    skipped_projects = 0
    failed_api_calls = 0

    existing_pairs = set()
    if not existing_df.empty:
        existing_pairs = set(
            zip(existing_df["project_name"].astype(str), existing_df["view"].astype(str), strict=False)
        )

    for _, row in projects_df.iterrows():
        expected_views = {f"{family}_{i}" for family in args.views for i in range(1, args.num_variants_per_view + 1)}
        if args.resume and not args.overwrite:
            existing_views = {view for project, view in existing_pairs if project == row["project_name"]}
            if expected_views.issubset(existing_views):
                skipped_projects += 1
                continue
        try:
            records, _ = generate_variants_for_row(row=row, args=args, raw_dir=raw_dir)
            generated_rows.extend(records)
        except Exception as exc:  # pragma: no cover - API/runtime dependent
            failed_api_calls += 1
            if args.debug:
                print(f"Failed to generate variants for {row['project_name']}: {exc}")

    output_columns = [
        "project_name",
        "project_description",
        "view",
        "view_family",
        "source",
        "model",
        "generation_batch_id",
    ]
    new_df = pd.DataFrame(base_rows + generated_rows, columns=output_columns)
    if args.resume and not existing_df.empty and not args.overwrite:
        non_base_existing = existing_df[existing_df["view"] != "base"].copy()
        combined = pd.concat(
            [
                pd.DataFrame(base_rows, columns=output_columns),
                non_base_existing,
                pd.DataFrame(generated_rows, columns=output_columns),
            ],
            ignore_index=True,
        )
        output_df = combined.drop_duplicates(subset=["project_name", "view"], keep="last")
    else:
        output_df = new_df.drop_duplicates(subset=["project_name", "view"], keep="last")

    output_df = output_df.sort_values(["project_name", "view"]).reset_index(drop=True)
    by_view_df = output_df.copy()

    write_csv(output_df, raw_output_path, overwrite=True)
    write_csv(by_view_df, by_view_output_path, overwrite=True)

    summary = {
        "number_of_input_projects": int(len(projects_df)),
        "number_of_generated_rows": int(len(output_df)),
        "number_of_rows_per_view": by_view_df["view_family"].value_counts().to_dict(),
        "number_of_failed_api_calls": failed_api_calls,
        "number_of_skipped_projects_due_to_resume": skipped_projects,
        "output_file_paths": {
            "project_variants_raw": str(raw_output_path),
            "project_variants_by_view": str(by_view_output_path),
            "raw_llm_dir": str(raw_dir),
        },
    }
    write_json(summary, summary_path, overwrite=True)
    print_terminal_summary(
        "Project Variant Generation",
        [
            f"input projects: {len(projects_df)}",
            f"output rows: {len(output_df)}",
            f"failed API calls: {failed_api_calls}",
            f"resume skips: {skipped_projects}",
            f"raw CSV: {raw_output_path}",
        ],
    )


if __name__ == "__main__":
    main()
