# RAG Evaluation Studio

An evaluation platform for Retrieval-Augmented Generation (RAG) pipelines.

**This is not another RAG chatbot.** It's a tool for measuring and comparing
how well RAG configurations retrieve and generate — retrieval quality,
faithfulness, hallucination, latency, and cost — so you can pick the
configuration that actually performs best on your data.

## Why

Most RAG demos show that a pipeline *works*. This project shows how well it
works, and lets you compare configurations (chunk size, embedding model,
vector store, Top-K, reranker, LLM) against each other with real metrics
instead of gut feel.

## Core capabilities

| Area | What it does |
|---|---|
| **Dataset Builder** | Upload PDFs/CSV/JSON or synthesize an eval set (question, ground truth, expected context/chunk) via an LLM, with manual editing |
| **Retrieval Evaluation** | Precision@K, Recall@K, MRR, Hit Rate, nDCG |
| **LLM Evaluation** | Faithfulness, Answer Relevancy, Context Precision/Recall, Hallucination, Answer Correctness — each with a plain-language explanation |
| **Experiment Manager** | Run and label multiple configurations, compare them side by side |
| **Dashboard** | Overall score, metric cards, radar chart, latency/cost graphs, best/worst questions |
| **Reports** | Export to PDF, CSV, JSON, Markdown |
| **Error Analysis** | Per-failure breakdown: question, ground truth, retrieved context, generated answer, scores, likely cause, suggested fix |

## Architecture

```
rag-evaluation-studio/
├── app/                    # Streamlit frontend
│   ├── main.py             # Entry point: page config, theme, nav shell
│   ├── components/         # Reusable UI: cards, styles, charts
│   └── pages/               # dataset_builder, retrieval_evaluation, llm_evaluation,
│                             # experiment_manager, dashboard, reports, settings
├── backend/
│   ├── config/              # Settings (pydantic), design tokens, LLM pricing table
│   ├── dataset/              # Chunking, loaders, synthesis, schema, storage
│   ├── retrieval/            # Metrics, TF-IDF/embedding retrievers, vector stores, reranker
│   ├── evaluation/
│   │   └── metrics/           # Faithfulness, relevancy, hallucination, etc. + LLM judge
│   ├── experiments/           # Config, runner, results schema, storage
│   ├── reports/                # PDF/CSV/JSON/Markdown report generation
│   └── utils/                  # Logging, unified LLM client (with offline fallback)
├── tests/                   # pytest suite, mirrors backend/ structure (87+ tests)
├── assets/                  # Static images/css
├── docs/                    # Design notes, roadmap
├── .github/workflows/       # CI: runs the test suite on push/PR
├── .streamlit/config.toml   # Native Streamlit theme (light, matches design tokens)
├── .env.example
├── pyproject.toml           # pytest + ruff config
├── LICENSE
└── requirements.txt
```

Design principle: **no giant single file**. The frontend never computes a
metric — it calls into `backend/`, which is plain, testable, framework-free
Python.

## Tech stack

- **Frontend:** Streamlit, Plotly, streamlit-aggrid
- **RAG/retrieval:** LangChain-compatible chunking, FAISS, ChromaDB, SentenceTransformers (each with graceful offline fallback)
- **Evaluation:** custom LLM-judge metrics with an offline heuristic fallback (`backend/evaluation/`); Ragas/DeepEval ship in `requirements.txt` as optional extensions for teams that want to swap in those frameworks' scorers
- **LLM providers:** OpenAI, Google Gemini, or a built-in offline heuristic provider — no key required to run the full app
- **Data:** Pandas, NumPy

## Getting started

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env               # then fill in API keys if you plan to run evaluations
streamlit run app/main.py
```

No API keys are required to launch the app shell — they're only needed once
you run an evaluation against a real LLM provider.

## Development roadmap

This project is being built in phases; each phase is fully functional before
the next begins.

- [x] **Phase 1** — Project structure, config, theming, app shell
- [x] **Phase 2** — Dataset Builder
- [x] **Phase 3** — Retrieval metrics
- [x] **Phase 4** — LLM evaluation metrics
- [x] **Phase 5** — Dashboard
- [x] **Phase 6** — Experiment comparison
- [x] **Phase 7** — Reporting
- [x] **Phase 8** — Testing and polishing

See `docs/roadmap.md` for phase-by-phase detail.

## Runs with zero configuration

Every feature works with no API key and no external services:

- **No LLM key configured** → an offline heuristic provider generates
  extractive answers and scores every metric with token-overlap
  heuristics instead of an LLM judge. The Settings page shows this
  status plainly; nothing fails silently.
- **`sentence-transformers` / `faiss-cpu` / `chromadb` not installed** →
  the embedding retriever and FAISS/Chroma vector stores fall back to
  the built-in TF-IDF retriever / in-memory numpy vector store, with a
  logged warning.

Add real keys in `.env` and install the optional packages any time to
get LLM-judged metrics and semantic retrieval — the UI and data model
don't change either way.

## Testing

87+ tests across chunking, retrieval metrics, the TF-IDF/vector-store
layer, all six LLM evaluation metrics, the experiment runner, storage
round-trips, and report generation.

```bash
pytest --cov=backend tests/
```

CI runs the full suite on every push via `.github/workflows/tests.yml`.

## License

MIT — see `LICENSE`.
