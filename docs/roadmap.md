# Development Roadmap

Each phase must be fully functional (runnable, no broken imports, no stub
buttons that error out) before the next phase begins.

## Phase 1 — Project structure + setup ✅
- Clean folder architecture (`app/`, `backend/`, `tests/`, `assets/`, `docs/`)
- Centralized `Settings` (pydantic) and design-token `theme.py`
- Loguru-based logging (`backend/utils/logger.py`)
- Streamlit app shell: page config, sidebar nav, light/dark toggle
- `requirements.txt`, `.env.example`, `.gitignore`, README

## Phase 2 — Dataset Builder ✅
- Upload PDF / CSV / JSON
- LLM-assisted synthetic Q&A generation (question, ground truth, expected
  context, expected chunk)
- Manual review/edit UI, save to `data/datasets/`

## Phase 3 — Retrieval metrics ✅
- `backend/retrieval/`: Precision@K, Recall@K, MRR, Hit Rate, nDCG
- Retrieval evaluation page: tables + bar/ranking charts

## Phase 4 — LLM evaluation ✅
- `backend/evaluation/metrics/`: Faithfulness, Answer Relevancy, Context
  Precision/Recall, Hallucination, Answer Correctness — custom LLM-judge
  prompts (`judge.py`) with an offline token-overlap heuristic fallback
  when no provider key is configured
- Explainability panel per metric (what/why/how)

## Phase 5 — Dashboard ✅
- Overall score, metric cards, radar chart, latency/cost graphs
- Best/worst performing questions table

## Phase 6 — Experiment comparison ✅
- `backend/experiments/`: define + run configurations (chunk size,
  embedding model, vector store, Top-K, reranker, LLM)
- Side-by-side comparison view, winner highlighting

## Phase 7 — Reporting ✅
- `backend/reports/generator.py`: CSV, JSON, Markdown, PDF export for a
  single experiment run, plus CSV/JSON/Markdown for multi-run comparisons
- `app/pages/reports.py`: pick a saved run (or several), preview, download
- PDF export uses `fpdf2`, imported lazily so the rest of the app stays
  importable without it installed; the page disables the PDF button with
  a clear hint if the package is missing

## Phase 8 — Testing and polishing ✅
- 87+ tests across `tests/backend/{dataset,retrieval,evaluation,experiments,reports}`
- `app/pages/settings.py`: masked API-key status, session-level provider/
  model/chunking defaults, app info — the last placeholder nav item
- `pyproject.toml` (pytest + ruff config), `.github/workflows/tests.yml` (CI),
  `LICENSE`
- Every page has an explicit empty state (no dataset yet, no runs yet) and
  wraps risky operations (chunking, generation, PDF export) in try/except
  with a user-facing `st.error`/`st.warning` rather than an unhandled crash
- README finalized: architecture, zero-config operation, testing, roadmap
