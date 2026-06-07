# Reformulation

This module builds and validates text-side query reformulations for the retrieval pipeline.

It covers two branches:
- `generate_project_variants.py` and `validate_project_variants.py` for project-description variants
- `generate_skill_variants.py` and `validate_skill_variants.py` for skill-keyword variants

## Inputs

Expected under `--data-root`:
- `project_descriptions.csv`
- `gpt5_skills.csv` or `skill_gpt5.csv`

## Outputs

Saved under `--out-root`:
- `processed/` for generated and validated CSVs
- `processed/llm_raw/` for raw model responses
- `summaries/` for stage summaries

Key files:
- `processed/project_variants_raw.csv`
- `processed/project_variants_by_view.csv`
- `processed/expanded_skills.csv`
- `processed/combined_skill_variants.csv`
- `processed/validated_project_variants.csv`
- `processed/rejected_project_variants.csv`
- `processed/validated_skill_variants.csv`
- `processed/rejected_skill_variants.csv`

## Prompts

Editable prompt templates live in `reformulation/prompts/`.

## Run

Full pipeline:

```bash
python reformulation/run_reformulation_validation_pipeline.py \
  --data-root data/kaito/raw \
  --out-root data/kaito \
  --model gpt-5 \
  --seed 42 \
  --debug
```

Example single stage:

```bash
python reformulation/generate_project_variants.py \
  --data-root data/kaito/raw \
  --out-root data/kaito \
  --model gpt-5 \
  --seed 42 \
  --debug
```

## Notes

- Generation uses strict JSON LLM outputs, then writes normalized CSVs in Python.
- Validation uses embedding checks plus LLM judging.
- Project validation also runs an independent view-family classifier.
- Resume logic is supported for generation; use `--overwrite` to replace existing outputs.
