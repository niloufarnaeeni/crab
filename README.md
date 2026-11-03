
# CRAB: Benchmarking Creator Retrieval and Attention for Generative Search

This repository contains the datasets and code for CRAB (Creator Retrieval and Attention Benchmark), which studies creator retrieval in incentivized attention economies. CRAB combines structured creator--project data, attention signals, skill annotations, query reformulation, text-based ranking, and graph-based retrieval for project-to-creator matching.

## Data

The `data/` folder contains three benchmark collections:

* `cookie_fun`
* `giverep`
* `kaito`

Each collection follows the same high-level structure:

* `dataset/` contains structured source tables used for benchmark construction.
* `ranker_input/` contains text-retrieval inputs for ranking models.

### `dataset/`

This folder contains structured metadata used for graph construction, retrieval, and attention-aware modeling:

* **attention_scores.csv**: Creator-level attention signals.
* **creator_details.csv**: Creator metadata, project associations, profile fields, and engagement features.
* **creators.csv**: Canonical list of creators.
* **project_descriptions.csv**: Base project descriptions used for text-side query construction.
* **project_variants.csv**: Reformulated or expanded project-description variants.
* **skills.csv**: Extracted skill taxonomy.
* **gpt5_skills.csv**: LLM-enhanced skill annotations, including `id`, `keyword`, `level`, `categories`, and `one_sentence_definition`.
* **skill_variants.csv**: Reformulated skill-keyword variants used for text augmentation.

Together, these files define the structured data used across both text-based and graph-based retrieval pipelines.

### `ranker_input/`

* **ranker_corpus.jsonl**: Supervised ranking instances used as input for text-based ranking models.

## Modules

### `text_retrieval/`

Text-based creator retrieval using cross-encoder ranking models, including MiniLM (`cross-encoder/ms-marco-MiniLM-L-6-v2`) and BGE Reranker (`BAAI/bge-reranker-base`). This module supports training, inference, and evaluation for project-to-creator ranking.

### `graph_retrieval/`

Graph-based creator retrieval using representation learning and GNN-based models, including MetaPath2Vec, GCN, and GraphSAGE.

### `skill_extraction/`

GPT-based structured skill extraction pipeline for constructing and enriching the project skill taxonomy.

### `reformulation/`

Query reformulation and validation pipeline for the text-based retrieval view. It generates controlled project-description variants and skill-keyword variants, validates them using embedding-based and LLM-based checks, and produces processed outputs for downstream retrieval experiments.

## Main Commands

### Graph Retrieval

```bash
python graph_retrieval/main.py
```

### Text Retrieval

```bash
python text_retrieval/text_retrieval/train_eval.py \
  --base_config text_retrieval/text_retrieval/train/reranker/config/training_bert.yaml \
  --test_jsonl data/kaito/ranker_input/ranker_corpus.jsonl
```

### Skill Extraction

```bash
python skill_extraction/run_gpt.py \
  --csv data/kaito/dataset/rootdata_descriptions.csv \
  --out kaito
```

### Reformulation

```bash
python reformulation/run_reformulation_validation_pipeline.py \
  --data-root data/kaito/dataset \
  --out-root data/kaito \
  --model gpt-5 \
  --seed 42 \
  --debug
```

## Heterogeneous Context--Creator--Skill Graph

CRAB models the attention economy as a heterogeneous context--creator--skill graph. Projects represent retrieval contexts, creators represent candidate actors, and skills represent inferred expertise requirements. These nodes are connected through participation and requirement relations.
