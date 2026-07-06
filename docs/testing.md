# Testing Guide

This document describes the automated test suite structure, verification commands, and local validation methodologies.

---

## 1. Test Suite Structure

Shortify AI features a comprehensive suite of unit and integration tests located under the `tests/` directory:

| Test File | Target | Description |
|---|---|---|
| `test_orchestrator.py` | LangGraph workflow | Validates complete graph navigation and state mapping. |
| `test_project_cache.py` | Caching layer | Validates cache file reads/writes, structure directories, and bypass nodes. |
| `test_color_grading.py` | color.py / color_service.py | Verifies LUT loading, FFmpeg filter compilation, and grading properties. |
| `test_intelligent_ducking.py`| editor_service.py | Verifies ducking ranges, volume curves, and music overlay blending. |
| `test_motion.py` | motion.py / editor_service.py | Validates constant speed and speed-ramping curve math. |
| `test_relevance_scoring.py` | relevance_service.py | Validates Gemini segment scoring, relevance mapping, and filters. |
| `test_render_cancellation.py` | worker_service.py | Validates thread termination on rendering abort requests. |
| `test_subtitle_chunking.py` | caption_chunking.py | Verifies token chunk sizes and safe boundary split calculations. |
| `test_text_graphics.py` | text_presets.py / overlay | Validates preset styling, fonts, and animation overlaps. |
| `test_quality_features.py` | editor_service.py | Validates general processing functions (crops, scales, audio). |
| `test_edl_validation.py` | edl_validation_service.py | Tests time duration alignment and validation constraint checks. |

---

## 2. Running the Tests

To run the automated tests, ensure your virtual environment is active and `PYTHONPATH` is set to the project root:

```bash
# Activate virtual environment
.venv\Scripts\activate

# Run the entire test suite
$env:PYTHONPATH="."; .\.venv\Scripts\pytest

# Run a specific test file
$env:PYTHONPATH="."; .\.venv\Scripts\pytest tests/test_project_cache.py

# Run a specific test case
$env:PYTHONPATH="."; .\.venv\Scripts\pytest tests/test_project_cache.py -k "test_cache_directory_structure"
```

---

## 3. Mocking & Dependencies

Tests utilize `unittest.mock` and `pytest` fixtures (such as `tmp_path` and `monkeypatch`) to simulate:
- Gemini API key requests and responses
- Groq/Llama storyboard generation payloads
- Database engine configurations and transaction states
- Giphy and Pixabay request mock objects
