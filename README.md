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
│   └── pages/               # Feature pages (dataset builder, dashboard, ...)
├── backend/
│   ├── config/              # Settings (pydantic) + design tokens (theme.py)
│   ├── dataset/              # Dataset upload/synthesis logic
│   ├── retrieval/            # Retrieval metrics (Precision@K, MRR, nDCG, ...)
│   ├── evaluation/
│   │   └── metrics/           # Faithfulness, relevancy, hallucination, etc.
│   ├── experiments/           # Experiment run + comparison logic
│   ├── reports/                # PDF/CSV/JSON/Markdown report generation
│   └── utils/                  # Logging and shared helpers
├── tests/                   # pytest suite, mirrors backend/ structure
├── assets/                  # Static images/css
├── docs/                    # Design notes, roadmap
├── .streamlit/config.toml   # Native Streamlit theme (light, matches design tokens)
├── .env.example
└── requirements.txt
```

Design principle: **no giant single file**. The frontend never computes a
metric — it calls into `backend/`, which is plain, testable, framework-free
Python.

## Tech stack

- **Frontend:** Streamlit, Plotly, streamlit-aggrid
- **RAG/retrieval:** LangChain, FAISS, ChromaDB, SentenceTransformers
- **Evaluation:** Ragas, DeepEval
- **LLM providers:** OpenAI, Google Gemini (local models supported via config)
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
- [ ] **Phase 7** — Reporting
- [ ] **Phase 8** — Testing and polishing

See `docs/roadmap.md` for phase-by-phase detail.

## Testing

```bash
pytest --cov=backend tests/
```

## License

MIT
