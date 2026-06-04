# Backend Testing Guide

## 1. Environment

- Python environment: `conda activate new_rag`
- Working directory for all commands below: `cd backend`
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

All commands below assume:

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

### 3.6 Run full automated test suite

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
```

### Retrieval / RAG tests

```bash
python -m unittest discover -s tests/unit/services/retrieval -p "test_*.py"
python -m unittest tests.integration.test_enhanced_retrieval_service
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
