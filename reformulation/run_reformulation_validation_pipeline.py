"""Run the generation and validation stages for reformulation only."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--model", default="gpt-5")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-project-generation", action="store_true")
    parser.add_argument("--skip-skill-generation", action="store_true")
    parser.add_argument("--skip-project-validation", action="store_true")
    parser.add_argument("--skip-skill-validation", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def run_step(script_name: str, args: argparse.Namespace) -> None:
    command = [
        sys.executable,
        str(Path(__file__).resolve().parent / script_name),
        "--data-root",
        args.data_root,
        "--out-root",
        args.out_root,
        "--model",
        args.model,
        "--seed",
        str(args.seed),
    ]
    if args.debug:
        command.append("--debug")
    if args.overwrite:
        command.append("--overwrite")
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    if not args.skip_project_generation:
        run_step("generate_project_variants.py", args)
    if not args.skip_skill_generation:
        run_step("generate_skill_variants.py", args)
    if not args.skip_project_validation:
        run_step("validate_project_variants.py", args)
    if not args.skip_skill_validation:
        run_step("validate_skill_variants.py", args)


if __name__ == "__main__":
    main()
