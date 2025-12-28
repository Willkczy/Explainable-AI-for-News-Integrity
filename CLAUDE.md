# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Essential Commands

### Running the Application
```bash
# Primary method (recommended)
python run.py

# Direct streamlit command
streamlit run app/app.py
```

### Development Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env to add API keys: GEMINI_API_KEY (required), GROQ_API_KEY, PERPLEXITY_API_KEY
```

### Testing Individual Modules
Each core module has standalone test capability:
```bash
python src/classifier.py          # Test fake news classifier
python src/explainer.py           # Test LLM explainer
python src/extractor.py           # Test simple claim extractor
python src/extractor_claimify.py  # Test 3-stage claim extractor
python src/retriever.py           # Test ChromaDB retriever
python src/retriever_pg.py        # Test PostgreSQL retriever
python src/perplexity_fact_checker.py  # Test Perplexity fact-checking
```

### Data Setup (Required Before First Run)
The application requires pre-trained models and Wikipedia database not tracked in git:
```bash
# 1. Build Wikipedia vector database (creates data/chroma_db_wiki/)
jupyter notebook notebooks/Big_data_WikiDB.ipynb

# 2. Train or download RoBERTa model (creates models/checkpoint_roberta/)
jupyter notebook notebooks/fake_news_classification.ipynb
```

## Architecture Overview

### Five-Stage Analysis Pipeline
```
Article Input → Classification → Claim Extraction → Evidence Retrieval → Fact-Checking → Explanation
```

1. **Classification** (src/classifier.py): RoBERTa model detects FAKE/REAL with confidence scores
2. **Claim Extraction** (src/extractor.py or src/extractor_claimify.py): Extracts verifiable claims via Groq/Llama
3. **Evidence Retrieval** (src/retriever.py or src/retriever_pg.py): Semantic search over Wikipedia
4. **Fact-Checking** (src/perplexity_fact_checker.py): AI-powered verification with web search
5. **Explanation** (src/explainer.py): Gemini generates comprehensive analysis with verdicts

### Environment-Aware Configuration
The system auto-detects runtime environment using `K_SERVICE` env variable:

**Cloud Run (production)**:
- Model paths: `/mnt/gcs/models/` (GCS FUSE mounted)
- Database: PostgreSQL + pgvector via src/retriever_pg.py
- Sentence transformers loaded from GCS to avoid HuggingFace rate limits

**Local development**:
- Model paths: `./models/checkpoint_roberta`
- Database: ChromaDB via src/retriever.py (or PostgreSQL if configured)
- Sentence transformers downloaded from HuggingFace

All environment detection logic is in config/config.py with helper functions:
- `_get_default_model_path()`: Returns model path based on environment
- `_get_sentence_transformer_path()`: Returns embeddings model path

### Key Integration Points

**config/config.py**: Single source of truth for all configuration
- API keys, model paths, database settings
- Environment-aware defaults with fallbacks
- Toggles: USE_POSTGRES, DEFAULT_EXTRACTOR_MODE

**app/app.py**: Streamlit UI orchestrates entire pipeline
- Uses `@st.cache_resource` to cache models across reruns
- Sidebar for runtime configuration (API keys, extractor mode, retrieval settings)
- Executes all 5 stages sequentially with progress tracking
- Displays results with color-coded verdicts, claim cards, evidence sections

**src/__init__.py**: Exports all core modules for clean imports
```python
from src import FakeNewsDetector, LLMExplainer, ClaimExtractor, WikiRetriever, etc.
```

### Dual Implementation Patterns

**Claim Extraction**: Two modes selectable at runtime
- **Simple** (src/extractor.py): Single prompt, fast, good for most articles
- **Claimify** (src/extractor_claimify.py): 3-stage pipeline (prefilter → extract → deduplicate) for higher quality

**Evidence Retrieval**: Two implementations with same interface
- **ChromaDB** (src/retriever.py): Local vector database, used in development
- **PostgreSQL** (src/retriever_pg.py): Cloud SQL with pgvector, used in production
- Toggle via `USE_POSTGRES` config setting

### External API Dependencies

All API integrations use environment variables for keys:

**Google Gemini** (required):
- Used in: src/explainer.py
- Model: gemini-2.5-flash
- Purpose: Generate comprehensive explanations with structured output (verdict, reasoning, key flags)
- API key: GEMINI_API_KEY

**Groq** (optional, improves quality):
- Used in: src/extractor.py, src/extractor_claimify.py
- Model: llama-3.1-8b-instant
- Purpose: Extract factual claims from article text
- API key: GROQ_API_KEY

**Perplexity** (optional):
- Used in: src/perplexity_fact_checker.py
- Model: llama-3.1-sonar-small-128k-online
- Purpose: Real-time web search fact-checking with citations
- API key: PERPLEXITY_API_KEY

### Cloud Deployment Architecture

**Google Cloud Run**:
- Containerized Streamlit app
- Auto-scaling based on traffic
- GCS FUSE volume mount for models at `/mnt/gcs/models/`

**Cloud SQL (PostgreSQL + pgvector)**:
- Fully managed vector database
- Connection via src/retriever_pg.py
- Credentials in POSTGRES_* env vars

**Cloud Storage (GCS)**:
- Stores large models (RoBERTa checkpoint, sentence transformers)
- Mounted as volume to avoid repeated downloads
- Bucket name: GCS_BUCKET_NAME config

## Important Implementation Details

### Model Loading
- FakeNewsDetector loads RoBERTa from local/GCS path (heavy, ~500MB)
- WikiRetriever/WikiRetrieverPG loads sentence-transformers (all-MiniLM-L6-v2)
- Both auto-detect environment and adjust paths accordingly
- Streamlit caches models using `@st.cache_resource` decorator

### Data Flow Through Pipeline
```python
# Typical execution in app/app.py:
label, confidence = detector.classify(text)
claims = extractor.extract(text, max_claims=10)
evidence = retriever.search_claims(claims, top_k=3)
fact_checks = checker.check_claims([c.text for c in claims])  # Optional
result = explainer.generate_explanation(
    title=title,
    text=text,
    classification=label,
    confidence=confidence,
    claims=claims,
    evidence=evidence,
    fact_check_results=fact_checks
)
```

### Explanation Output Structure
LLMExplainer returns dict with:
- `display_status`: Verdict string (False, Misleading, Unverified, Partially Verified, Verified)
- `explanation`: Detailed reasoning text
- `key_flags`: List of important indicators
- `claim_analysis`: Per-claim verification status (Supported/Contradicted/Unverified)

### Wikipedia Database
- ETL pipeline in notebooks/Big_data_WikiDB.ipynb processes Wikipedia dumps
- Creates vector embeddings using sentence-transformers
- Local: Stores in ChromaDB at data/chroma_db_wiki/
- Cloud: Stores in PostgreSQL table with pgvector extension
- Both support semantic search with cosine similarity

### Perplexity Fact-Checking
- Takes list of claim strings
- Returns list of dicts per claim: {claim, verdict, explanation, sources}
- Verdicts: TRUE, FALSE, PARTIALLY TRUE, UNVERIFIED
- Sources: List of URLs from web search results
- Heuristic analysis of search results to determine verdict

## File Locations

**Core business logic**: src/
- Each module is self-contained with class definition
- All have `__main__` block for standalone testing
- Import from src package: `from src import ClassName`

**Configuration**: config/config.py
- Modify settings here rather than hardcoding in modules
- Add new env vars to .env.example when adding config

**UI**: app/app.py
- All Streamlit code isolated here
- Uses modules from src/ package
- Manages user input, displays results

**Data/Models** (gitignored):
- data/chroma_db_wiki/: ChromaDB vector store
- models/checkpoint_roberta/: Fine-tuned RoBERTa model

**Notebooks** (development):
- notebooks/Big_data_WikiDB.ipynb: Wikipedia ETL and database creation
- notebooks/fake_news_classification.ipynb: Model training and evaluation
- notebooks/EDA_and_preprocessing.ipynb: Dataset exploration
