# Error Handling & Fault Tolerance

This document details the strategies and configurations implemented in Shortify AI to ensure robust performance under API rate limits, validation exceptions, and external service failures.

---

## 1. Google API Key Rate-Limit Handling (429 Backoff)

To protect the media analysis pipeline from Gemini rate limits (`429 Resource Exhausted`), the client manager incorporates an automatic backoff and selection system:

1. **Cooldown Tracking**: If an API key slot receives a 429 response, it is flagged with a timestamp. The selection logic automatically skips this slot for a designated cooldown period.
2. **Regex Timeout Delay Parsing**: The rotator uses regex patterns to capture the exact wait time returned in Google's error messages:
   - `please retry in X seconds`
   - `retryDelay: Y`
3. **Execution Sleep**: The pinned file execution thread sleeps for the exact duration specified (plus a small safety buffer) before retrying the call.

---

## 2. EDL Validation Failures (`EDL_VALIDATION_FAIL`)

The Creative Director agent compiles timeline edit decisions that are validated against strict structural rules (e.g. total duration alignment, speeds, etc.) inside `edl_validation_service.py`.

To handle edge cases where the AI director fails to formulate a compliant structure:
- **`stop` mode**: The pipeline aborts immediately, throwing an `EDLGenerationError`.
- **`pass` mode**: If the director fails to create a valid EDL after 3 attempts, the system intercepts the error, logs a warning, and proceeds to render the video using the last available EDL plan to prevent render failures.

This is controlled using the `EDL_VALIDATION_FAIL` environment variable in the `.env` file.

---

## 3. Caption Safety Check Self-Correction Loop

The system enforces TikTok/Reels screen layouts to ensure captions do not overlap app UI controls:

```
  Creative Director                  SubtitleAgent
+-------------------+             +------------------+
| Generate EDL Plan | ----------> | Validate Overlay |
+-------------------+             +--------+---------+
         ^                                 |
         |  Regenerate with feedback       | Failed Zone check
         +---------------------------------+
```

- **Safety Zone Checks**: Overlay coordinates (width/height on a `1080x1920` layout) are scanned. Overlay titles must not reside in the top 150px (header), right 120px (action items), or bottom 300px (app details/audio info).
- **Self-Correction Feedback Loop**: If an overlay fails safety validation, a detailed layout warning (e.g., `Text overlaps right buttons; shift X coordinate to the left`) is appended to `edl_feedback` in the orchestrator state. The graph triggers a loop back to `generate_edl` for the Creative Director to regenerate a revised plan. The loop runs for a maximum of 5 attempts.

---

## 4. Subprocess Execution Timeouts

All FFmpeg and OpenCV command executions are wrapped in Python `subprocess.run` executions with strict timeouts to prevent infinite thread blocking:
- **Stabilization Pass**: `timeout=180` (Pass 1) and `timeout=300` (Pass 2).
- **Clip Processing**: `timeout=300`.
- **Transitions and Merges**: `timeout=60`.

If a command times out or fails, exceptions are logged and intermediate files are cleaned up in a `finally` block to prevent resource leaks.
