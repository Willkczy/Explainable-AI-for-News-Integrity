# Perplexity Integration Guide

This project can optionally use the Perplexity Search API to fact-check extracted claims with live web search results and source URLs.

Perplexity is not required to run the core pipeline. When it is disabled or no API key is provided, the app still runs classification, claim extraction, Wikipedia retrieval, and Gemini explanation.

## Current Role In The Pipeline

Perplexity is used in step 4 of the analysis pipeline:

```text
Article -> Classification -> Claim Extraction -> Wikipedia Retrieval -> Perplexity Fact-Checking -> Gemini Explanation
```

The implementation is in `src/perplexity_fact_checker.py` and is called from `app/app.py`.

Current behavior:

- Uses the Perplexity Search API through the official Python SDK package `perplexityai`
- Imports the client with `from perplexity import Perplexity`
- Searches each extracted claim with a fact-check style query
- Requests 2 search results per claim
- Uses `max_tokens_per_page=1024`
- Applies a simple keyword heuristic to label results as `TRUE`, `FALSE`, `PARTIALLY TRUE`, or `UNVERIFIED`
- Passes the verdict, explanation, and source links into the Gemini explainer

Because the verdicting step is heuristic, Perplexity results should be treated as supporting evidence, not as a final authoritative truth label.

## Installation

Install dependencies from the project root:

```bash
pip install -r requirements.txt
```

If installing only the Perplexity SDK:

```bash
pip install perplexityai
```

The package name is `perplexityai`, but the Python import is:

```python
from perplexity import Perplexity
```

## API Key Setup

Add your API key to `.env`:

```bash
PERPLEXITY_API_KEY=your-perplexity-api-key-here
```

You can also paste the key into the Streamlit sidebar at runtime.

The app reads the key in this order:

1. Sidebar input in `app/app.py`
2. `PERPLEXITY_API_KEY` from the environment or `.env`

## Running In The App

Start the Streamlit app:

```bash
python run.py
```

Or:

```bash
streamlit run app/app.py
```

In the sidebar:

1. Enter `PERPLEXITY_API_KEY`, or rely on the value from `.env`
2. Enable `Enable Perplexity Fact Checking`
3. Run article analysis

Perplexity results appear under each extracted claim as fact-check results with source links.

## Standalone Test

Run the fact checker module directly:

```bash
python src/perplexity_fact_checker.py
```

If your environment uses `python3` instead of `python`:

```bash
python3 src/perplexity_fact_checker.py
```

## Programmatic Usage

```python
from src.perplexity_fact_checker import PerplexityFactChecker

checker = PerplexityFactChecker(api_key="your-perplexity-api-key")

results = checker.check_claims([
    "Water boils at 100 degrees Celsius at sea level.",
    "The Great Wall of China is visible from the moon."
])

for result in results:
    print(result["claim"])
    print(result["verdict"])
    print(result["explanation"])
    print(result["sources"])
```

## Output Format

Each result is a dictionary:

```json
{
  "claim": "The statement being checked",
  "verdict": "TRUE | FALSE | PARTIALLY TRUE | UNVERIFIED | ERROR",
  "explanation": "Short analysis generated from search snippets",
  "sources": [
    "Source Title - https://source-url.example"
  ],
  "detailed_sources": [
    {
      "title": "Source Title",
      "url": "https://source-url.example",
      "snippet": "Search result snippet"
    }
  ]
}
```

In the Streamlit pipeline, these results are converted into the format expected by `LLMExplainer`.

## Verdict Heuristic

The current implementation combines snippets from the top search results and checks for indicator words:

- `FALSE`: words such as `false`, `myth`, `debunk`, `not true`, `incorrect`, `wrong`, `fake`, `hoax`
- `TRUE`: words such as `true`, `correct`, `verified`, `confirm`, `accurate`, `fact`
- `PARTIALLY TRUE`: words such as `partially`, `partly`, `some truth`, `misleading`, `context`
- `UNVERIFIED`: used when the retrieved snippets do not strongly indicate one of the above

This is intentionally simple and should be improved before treating verdicts as production-grade fact-check labels.

## Troubleshooting

### Import Error

If you see an error that the Perplexity package is not available:

```bash
pip install perplexityai
```

Remember that the installed package is `perplexityai`, while the import path is `perplexity`.

### Missing API Key

Make sure one of these is true:

- `PERPLEXITY_API_KEY` is set in `.env`
- The key is entered in the Streamlit sidebar

### Fact-Checking Does Not Run

Check that:

- `Enable Perplexity Fact Checking` is enabled in the sidebar
- Claims were extracted successfully
- A Perplexity API key is available
- The Perplexity SDK is installed

### API Request Failed

Check:

- API key validity
- API credits or billing status
- Network connectivity
- Perplexity service status

## Related Files

- `src/perplexity_fact_checker.py`: Perplexity API wrapper and verdict heuristic
- `app/app.py`: Sidebar key input, toggle, and pipeline integration
- `src/explainer.py`: Incorporates Perplexity results into the final Gemini explanation
- `config/config.py`: Defines `PERPLEXITY_API_KEY`
- `.env.example`: Shows expected environment variable setup
