# Backend Testing Guide

## 1. Environment

- Python environment: `conda activate new_rag`
- Repository-level quality commands must run from the repository root.
- Backend-only test commands must run from `backend/`.
- Test framework: `unittest`

Most automated tests in this repository are designed to run offline with fake services, stubs, or temporary SQLite files.

## 2. Core testing rules

### 2.1 Tests that must not connect to real external services

The following automated tests must not call real external services:

- unit tests under `tests/unit/`
- api tests under `tests/api/`
- integration tests under `tests/integration/`
- golden smoke tests under `tests/golden/`
- agent-related tests under `tests/test_agent_*.py`

In automated test runs, do not connect to:

- real arXiv API
- real arXiv OAI network endpoints
- real LLM providers
- real embedding providers
- real rerank providers
- real Milvus / vector DB services
- real PDF download / Docling / PyMuPDF parsing backends

These tests should use existing fake helpers in `tests/helpers/`, mock objects, dynamic import stubs, or in-memory fake services.

### 2.2 Tests that may use temporary SQLite

The following tests are allowed to use temporary SQLite only:

- database service tests
- memory / recommendation integration tests
- paper QA integration tests
- QA index job flow tests
- any router or service test that explicitly uses `tests/helpers/sqlite.py`

Do not point automated tests at real project databases such as persisted recommendation or OAI DB files.

### 2.3 Manual-only external integration

Manual-only external integration checks are not part of automated tests and should not enter default CI, including:

- real arXiv search and download verification
- real model provider calls
- real embedding / rerank provider validation
- real Milvus integration
- end-to-end PDF parsing with external/runtime-heavy dependencies
- production-like environment smoke checks

## 3. How to run tests

Recommended repository-level quality gate:

```bash
python scripts/check_quality.py
```

The default gate runs basic doctor, backend static checks, backend automated tests, backend startup smoke tests, frontend tests, and frontend build checks, then prints a final summary. It is the preferred command before submitting code because it keeps the offline-safe backend and frontend checks in one place.

CI uses the same staged gate through `python scripts/check_quality.py ci`. Codex changes should report the exact checks that were run, their pass/fail status, and any environment-related blockers; see [Maintenance and Quality](maintenance-quality.md).

Backend static checks are not ordinary unit tests. They run `compileall`, import smoke checks for key app/router/service/agent/tool/config modules, and optional low-noise `ruff` rules before the heavier test suite starts.

Backend startup smoke tests are also separate from full integration tests. They create the FastAPI app in lazy mode, enter the lifespan through `TestClient`, assert core routers and stable error payloads, and use fake services so no real LLM, Embedding, Milvus, arXiv, rerank, or PDF parsing backend is contacted.

Useful staged entrypoints from the repository root:

```bash
python scripts/check_quality.py backend
python scripts/check_quality.py frontend
python scripts/check_quality.py smoke
python scripts/check_quality.py static
python scripts/check_quality.py compile
python scripts/check_quality.py backend-startup-smoke
```

`static` is the recommended backend static layer. `compile` only runs the narrower Python compilation check.

Backend-only commands below assume:

```bash
conda activate new_rag
cd backend
```

### 3.1 Run existing test suite discovery

```bash
python -m unittest discover -s tests -p "test_*.py"
```

### 3.2 Run unit tests

```bash
python -m unittest discover -s tests/unit -p "test_*.py"
```

### 3.3 Run API tests

```bash
python -m unittest discover -s tests/api -p "test_*.py"
```

### 3.4 Run integration tests

```bash
python -m unittest discover -s tests/integration -p "test_*.py"
```

### 3.5 Run golden smoke tests

```bash
python -m unittest tests.golden.test_rag_golden_smoke
```

### 3.6 Run RAG golden pipeline tests

```bash
python -m pytest tests/golden/test_rag_golden_pipeline.py
```

This suite exercises the real `EnhancedRetrievalService` orchestration with fake embeddings, an in-memory vector store, fixed chunks, and temporary trace output. It validates evidence hit-at-k, key retrieval debug stages, trace export, and figure/table evidence flow without calling a real model or Milvus.

### 3.7 Run full automated test suite

```bash
python -m unittest discover -s tests -p "test_*.py"
```

This includes unit, api, integration, helper/infrastructure, and golden smoke tests that follow the `test_*.py` naming rule.

## 4. Frequently used targeted commands

### Agent tests

```bash
python -m unittest discover -s tests -p "test_agent*.py"
python -m unittest tests.api.test_agent_router
python -m unittest tests.integration.test_agent_chat_flow
python -m pytest tests/api/test_agent_router.py
```

### Retrieval / RAG tests

```bash
python -m unittest discover -s tests/unit/services/retrieval -p "test_*.py"
python -m unittest tests.integration.test_enhanced_retrieval_service
python -m pytest tests/golden/test_rag_golden_pipeline.py
```

### Paper QA tests

```bash
python -m unittest tests.integration.test_paper_qa_service
python -m unittest tests.integration.test_qa_index_build_flow
```

### Memory / Recommendation / Intent tests

```bash
python -m unittest tests.unit.services.intent.test_intent_service
python -m unittest tests.integration.test_memory_service
python -m unittest tests.integration.test_recommendation_flow
```

The memory integration suite also covers the Agent session-memory read/write loop: final Agent state is reduced to a small persisted memory patch, then reloaded and merged with frontend context. This catches regressions where Agent state, selected paper context, or tool-call summaries stop surviving across turns.

### Database service tests

```bash
python -m unittest discover -s tests -p "test_database*.py"
```

## 5. Coverage commands

Run coverage across the automated test suite:

```bash
python -m coverage run -m unittest discover -s tests -p "test_*.py"
python -m coverage report -m
```

Optional HTML output:

```bash
python -m coverage html
```

Then open `htmlcov/index.html` locally.

## 6. Coverage acceptance notes

- Coverage is used as a regression visibility tool, not as permission to connect to external services.
- A coverage run must remain offline-safe.
- If a test requires real upstream services, move it to a separate manual script rather than adding it to default `unittest` discovery.
- Prefer focused fake-based tests over broad unstable integration coverage.

## 7. Existing helper locations

Common reusable test helpers live under:

- `tests/helpers/sqlite.py`
- `tests/helpers/fake_embedding_service.py`
- `tests/helpers/fake_generation_service.py`
- `tests/helpers/fake_vector_store_service.py`
- `tests/helpers/fake_arxiv_service.py`
- `tests/helpers/fake_paper_qa_service.py`
- `tests/helpers/retrieval.py`
- `tests/helpers/agent_runtime.py`

Use these helpers before introducing new heavy mocks.

## 8. Troubleshooting

### Import or dependency errors

- Confirm the environment is `conda activate new_rag`
- Run commands from `backend/`
- Prefer existing test helpers that stub optional runtime dependencies

### Unexpected real network or model calls

- Check whether a test forgot to override dependencies
- Verify fake service injection or `sys.modules` stubs are active
- Do not accept a fix that passes only because real external services happen to be reachable

### SQLite pollution concerns

- Use temporary SQLite helpers from `tests/helpers/sqlite.py`
- Do not reuse real repository DB files in automated tests
