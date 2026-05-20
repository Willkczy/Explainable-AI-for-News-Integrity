# AGENTS.md

This file gives Codex quick working context for this repository. Keep this file short and operational; use the project docs below as the source of truth for detailed architecture and setup.

## Source Of Truth

- `README.md`: project overview, local setup, usage, links
- `ARCHITECTURE.md`: pipeline, runtime configuration, local/cloud differences
- `CONTRIBUTING.md`: Git workflow, module ownership, contributor setup
- `PERPLEXITY_SETUP.md`: Perplexity Search API integration
- `.env.example`: expected environment variables

When implementation and docs disagree, inspect the code first, then update the relevant source document instead of adding another copy of the same explanation here.

## Essential Commands

```bash
# Run the Streamlit app
python run.py
streamlit run app/app.py

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env

# Local ChromaDB development
# Keep USE_POSTGRES=false in .env unless PostgreSQL credentials are configured.

# Module smoke tests
python src/classifier.py
python src/explainer.py
python src/extractor.py
python src/extractor_claimify.py
python src/retriever.py
python src/retriever_pg.py
python src/perplexity_fact_checker.py
```

Some smoke tests require local model/database files or API keys.

## Runtime Requirements

- Local development is configured for Python 3.12.
- The current `Dockerfile` uses `python:3.11` for Cloud Run.
- The RoBERTa checkpoint and Wikipedia vector database are not tracked in git.
- Local ChromaDB data is expected at `data/chroma_db_wiki/`.
- The classifier model is expected at `models/checkpoint_roberta/`.
- Cloud Run uses GCS-mounted model paths under `/mnt/gcs/models/`.

## Pipeline Summary

```text
Article Input
-> Classification (src/classifier.py)
-> Claim Extraction (src/extractor.py or src/extractor_claimify.py)
-> Evidence Retrieval (src/retriever.py or src/retriever_pg.py)
-> Optional Perplexity Fact-Checking (src/perplexity_fact_checker.py)
-> Gemini Explanation (src/explainer.py)
```

`app/app.py` is the Streamlit orchestrator. It normalizes extracted claims into plain strings before retrieval and explanation.

## Configuration Notes

- `config/config.py` is the central configuration module.
- `GEMINI_API_KEY` enables Gemini explanations.
- `GROQ_API_KEY` enables Simple/Claimify claim extraction.
- `PERPLEXITY_API_KEY` enables optional Perplexity fact-checking.
- `GOOGLE_FACTCHECK_API_KEY` is legacy and currently unused by the Streamlit app.
- Local development should usually set `USE_POSTGRES=false`.
- Cloud SQL/PostgreSQL retrieval requires `USE_POSTGRES=true` and valid `POSTGRES_*` values.
- `MAX_CLAIMS_PER_ARTICLE` exists in config, but `app/app.py` currently passes `max_claims=5` directly to both extractor modes.

## Core Modules

- `src/classifier.py`: RoBERTa fake/real classifier
- `src/explainer.py`: Gemini explanation generator
- `src/extractor.py`: simple Groq-based claim extractor
- `src/extractor_claimify.py`: 3-stage Claimify extractor
- `src/retriever.py`: local ChromaDB Wikipedia retriever
- `src/retriever_pg.py`: PostgreSQL + pgvector retriever
- `src/perplexity_fact_checker.py`: Perplexity Search API wrapper and heuristic verdicting
- `app/app.py`: Streamlit UI and pipeline orchestration
- `config/config.py`: environment-aware settings

## Working Guidelines

- Keep code changes scoped to the requested module or document.
- Prefer existing project patterns over new abstractions.
- Update `.env.example` whenever adding or changing environment variables.
- Update `README.md` for onboarding changes and `ARCHITECTURE.md` for pipeline/runtime changes.
- Avoid committing generated data, model checkpoints, `.env`, or local database files.
- Treat Perplexity verdicts as supporting evidence; current verdicting is heuristic, not authoritative.
- If a test cannot run because local models, databases, or API keys are missing, say that explicitly.
