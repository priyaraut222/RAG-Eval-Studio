# Development Roadmap

Each phase must be fully functional (runnable, no broken imports, no stub
buttons that error out) before the next phase begins.

## Phase 1 — Project structure + setup ✅
- Clean folder architecture (`app/`, `backend/`, `tests/`, `assets/`, `docs/`)
- Centralized `Settings` (pydantic) and design-token `theme.py`
- Loguru-based logging (`backend/utils/logger.py`)
- Streamlit app shell: page config, sidebar nav, light/dark toggle
- `requirements.txt`, `.env.example`, `.gitignore`, README

## Phase 2 — Dataset Builder
- Upload PDF / CSV / JSON
- LLM-assisted synthetic Q&A generation (question, ground truth, expected
  context, expected chunk)
- Manual review/edit UI, save to `data/datasets/`

## Phase 3 — Retrieval metrics
- `backend/retrieval/`: Precision@K, Recall@K, MRR, Hit Rate, nDCG
- Retrieval evaluation page: tables + bar/ranking charts

## Phase 4 — LLM evaluation
- `backend/evaluation/metrics/`: Faithfulness, Answer Relevancy, Context
  Precision/Recall, Hallucination, Answer Correctness (via Ragas/DeepEval)
- Explainability panel per metric (what/why/how)

## Phase 5 — Dashboard
- Overall score, metric cards, radar chart, latency/cost graphs
- Best/worst performing questions table

## Phase 6 — Experiment comparison
- `backend/experiments/`: define + run configurations (chunk size,
  embedding model, vector store, Top-K, reranker, LLM)
- Side-by-side comparison view, winner highlighting

## Phase 7 — Reporting
- `backend/reports/`: PDF, CSV, JSON, Markdown export

## Phase 8 — Testing and polishing
- `tests/` coverage for backend metrics and dataset logic
- UI polish pass, empty/error states, README finalization
