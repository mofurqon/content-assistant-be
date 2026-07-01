# Content Assistant API — Technical Documentation

Base URL: `http://localhost:8000`  
Interactive docs: `http://localhost:8000/docs`  
Version: `3.0`

---

## Overview

The API uses a **POST-to-trigger, GET-to-stream** pattern backed by a stateful LangGraph pipeline.

- **POST** endpoints register an action and return `{ session_id }` immediately (no body streaming).
- **GET** endpoints open an SSE stream and drive graph execution for that session.

This separation lets the React frontend use the browser's native `EventSource` API on GET endpoints, while POST calls are plain `fetch` requests.

### Full call sequence

```
POST /v1/ideas                          → { session_id }
GET  /v1/ideas/{session_id}/stream      → SSE: tokens … interrupted (ideas)

POST /v1/pipeline/{session_id}          → { session_id }
GET  /v1/pipeline/{session_id}/stream   → SSE: tokens … awaiting_feedback (draft)

POST /v1/pipeline/{session_id}/resume   → { session_id }
GET  /v1/pipeline/{session_id}/stream   → SSE: tokens … done (final result)
```

State is held in-memory by LangGraph's `MemorySaver`, keyed by `session_id`. Sessions do not survive server restarts.

---

## Authentication

No authentication required. CORS is open to all origins (`*`).

---

## Common Types

### `SessionResponse`

Returned by all POST endpoints.

| Field | Type | Description |
|---|---|---|
| `session_id` | string | UUID identifying the LangGraph session |

### `ContentCriteria`

Optional object accepted by `POST /v1/pipeline/{session_id}`. All fields have defaults and can be omitted entirely.

| Field | Type | Default | Description |
|---|---|---|---|
| `target_audience` | string | `"general readers"` | Who the article is written for |
| `content_type` | string | `"Article"` | Format of the output (e.g. Article, Blog Post, Tutorial) |
| `tone` | string | `"Professional and informative"` | Writing tone |

### `EvalResult`

| Field | Type | Description |
|---|---|---|
| `scores` | `{ [criterion]: int }` | Per-criterion score (1–5). Criteria: Clarity, Relevance, Completeness, Accuracy, Actionability, Retrieval Relevance |
| `average` | float | Mean of all criterion scores |
| `reasoning` | string | LLM-generated explanation of the scores |

### `ResearchResult`

| Field | Type | Description |
|---|---|---|
| `queries` | string[] | Search queries generated from the selected idea |
| `summary` | string | Synthesised summary of web research findings |

### SSE Event Types

All GET stream endpoints emit newline-delimited SSE events:

```
data: <JSON payload>\n\n
```

| `type` | Fields | Emitted by |
|---|---|---|
| `token` | `content: string` | Both GET stream endpoints — each LLM output token |
| `node` | `name: string` | Both GET stream endpoints — when a pipeline node begins |
| `interrupted` | `ideas: string[]` | `GET /v1/ideas/{session_id}/stream` — graph paused awaiting idea selection |
| `awaiting_feedback` | `draft: string` | `GET /v1/pipeline/{session_id}/stream` (first call) — graph paused awaiting human feedback |
| `done` | `result: PipelineResult` | `GET /v1/pipeline/{session_id}/stream` (second call) — pipeline complete |

**`PipelineResult`** (carried in `done.result`):

| Field | Type | Description |
|---|---|---|
| `idea` | string | The selected article idea |
| `draft` | string | Initial generated draft |
| `article` | string | Final article combining improved draft + research |
| `image_prompt` | string | Suggested image generation prompt |
| `evaluations` | `EvalResult[]` | One entry per evaluation pass (up to 3) |
| `research` | `ResearchResult` | Web research queries and summary |

---

## Endpoints

### `GET /health`

Health check.

**Response `200`**
```json
{ "status": "ok" }
```

---

### `POST /v1/ideas`

Create a new session for the given topic. The session is initialised but execution does not start until the client opens the stream.

**Request**
```json
{ "topic": "software testing best practices" }
```

| Field | Type | Required | Description |
|---|---|---|---|
| `topic` | string | yes | Free-text topic the user wants to write about |

**Response `200`**
```json
{ "session_id": "a1b2c3d4-e5f6-..." }
```

---

### `GET /v1/ideas/{session_id}/stream` — SSE

Start idea generation for the session. Streams LLM tokens while ideas are being generated, then emits a final `interrupted` event and pauses.

**Path parameter:** `session_id` — from `POST /v1/ideas`

**Response `200`** — `Content-Type: text/event-stream`

```
data: {"type": "token", "content": "1."}

data: {"type": "token", "content": " Why Shift-Left..."}

...

data: {"type": "interrupted", "ideas": ["Why Shift-Left Testing Reduces Production Bugs", "A Practical Guide to the Test Pyramid", "..."]}
```

**Error `404`** — session not found or this endpoint was already called for that session.

---

### `POST /v1/pipeline/{session_id}`

Stage the user's selected idea and optional writing criteria. Execution does not resume until the client opens the stream.

**Path parameter:** `session_id` — from `POST /v1/ideas`

**Request**
```json
{
  "selected_idea": "Why Shift-Left Testing Reduces Production Bugs",
  "criteria": {
    "target_audience": "QA engineers and engineering managers",
    "content_type": "Article",
    "tone": "Professional and practical"
  }
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `selected_idea` | string | yes | The idea chosen by the user |
| `criteria` | `ContentCriteria` | no | Writing criteria; all sub-fields default if omitted |

**Response `200`**
```json
{ "session_id": "a1b2c3d4-e5f6-..." }
```

---

### `GET /v1/pipeline/{session_id}/stream` — SSE

Resume the pipeline from its current pause point. Used **twice** across the full flow:

- **First call** — resumes from idea selection, runs retrieve → generate, then pauses. Emits `awaiting_feedback` with the draft.
- **Second call** — resumes from draft review, runs evaluate → improve → research → finalize. Emits `done` with the complete result.

**Path parameter:** `session_id`

**Response `200`** — `Content-Type: text/event-stream`

*First call (draft phase):*
```
data: {"type": "node", "name": "retrieve"}

data: {"type": "node", "name": "generate"}

data: {"type": "token", "content": "## Introduction\n"}

data: {"type": "token", "content": "Shift-left testing..."}

...

data: {"type": "awaiting_feedback", "draft": "## Introduction\nShift-left testing..."}
```

*Second call (finalize phase):*
```
data: {"type": "node", "name": "evaluate"}

data: {"type": "node", "name": "improve"}

data: {"type": "node", "name": "evaluate"}

data: {"type": "node", "name": "research"}

data: {"type": "node", "name": "finalize"}

data: {"type": "token", "content": "# Why Shift-Left Testing..."}

...

data: {"type": "done", "result": {
  "idea": "Why Shift-Left Testing Reduces Production Bugs",
  "draft": "## Introduction\n...",
  "article": "# Why Shift-Left Testing Reduces Production Bugs\n\n...",
  "image_prompt": "A developer reviewing automated test results on a large monitor...",
  "evaluations": [
    {
      "scores": {"Clarity": 4, "Relevance": 5, "Completeness": 3, "Accuracy": 4, "Actionability": 4, "Retrieval Relevance": 4},
      "average": 4.0,
      "reasoning": "The draft covers the core concept well but lacks depth on tooling..."
    }
  ],
  "research": {
    "queries": ["shift-left testing benefits 2024", "CI/CD test automation strategies"],
    "summary": "Recent industry data shows teams adopting shift-left practices..."
  }
}}
```

**Pipeline node sequence**
```
retrieve → generate → [pause: awaiting_feedback] → evaluate → [improve → evaluate]* → research → finalize
```

The `improve → evaluate` cycle runs at most twice and stops early when the average score reaches ≥ 4.0.

---

### `POST /v1/pipeline/{session_id}/resume`

Stage human feedback for the improvement loop. Execution does not resume until the client opens the stream again.

**Path parameter:** `session_id`

**Request**
```json
{ "human_feedback": "Add a section on CI/CD integration and shorten the intro" }
```

| Field | Type | Required | Description |
|---|---|---|---|
| `human_feedback` | string | no | Reviewer notes folded into the first improvement pass alongside LLM self-eval scores. Defaults to `""` |

**Response `200`**
```json
{ "session_id": "a1b2c3d4-e5f6-..." }
```

---

## Client Integration Guide

### Step 1 — Create a session

```bash
curl -X POST http://localhost:8000/v1/ideas \
  -H "Content-Type: application/json" \
  -d '{"topic": "software testing best practices"}'
# → { "session_id": "a1b2c3d4-..." }
```

### Step 2 — Stream idea generation

```bash
curl -N http://localhost:8000/v1/ideas/a1b2c3d4-.../stream
```

Read events until `interrupted`. Extract `ideas` and display them to the user.

### Step 3 — Stage the selected idea

```bash
curl -X POST http://localhost:8000/v1/pipeline/a1b2c3d4-... \
  -H "Content-Type: application/json" \
  -d '{
    "selected_idea": "Why Shift-Left Testing Reduces Production Bugs",
    "criteria": { "target_audience": "QA engineers", "tone": "Professional and practical" }
  }'
```

### Step 4 — Stream the draft

```bash
curl -N http://localhost:8000/v1/pipeline/a1b2c3d4-.../stream
```

Stream `token` events to render the draft. Wait for the `awaiting_feedback` event, then show the draft to the user and collect their feedback.

### Step 5 — Stage human feedback

```bash
curl -X POST http://localhost:8000/v1/pipeline/a1b2c3d4-.../resume \
  -H "Content-Type: application/json" \
  -d '{"human_feedback": "Include examples from the testing pyramid"}'
```

### Step 6 — Stream the final article

```bash
curl -N http://localhost:8000/v1/pipeline/a1b2c3d4-.../stream
```

Stream `token` events to render the article. Consume the `done` event for the complete structured result.

---

## Improvement Loop Behaviour

After the draft is reviewed, the pipeline runs a self-evaluation step. The LLM scores the draft on six criteria on a 1–5 scale.

- If the average score is **below 4.0** and fewer than **2 improvement iterations** have run, the draft is rewritten and re-evaluated.
- `human_feedback` is incorporated into the **first** improvement pass only.
- The `evaluations` array in the `done` result contains one entry per evaluation pass — up to three.

---

## Error Handling

FastAPI returns standard HTTP error responses. Validation errors return `422 Unprocessable Entity`:

```json
{
  "detail": [
    {
      "loc": ["body", "selected_idea"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```

| Scenario | Status |
|---|---|
| `GET /v1/ideas/{session_id}/stream` with unknown or already-consumed session | `404` |
| Missing required request field | `422` |
| LLM or Firecrawl failure | `500` |
| Session not found in LangGraph (e.g. server restarted) | `500` — start a new session |

---

## Running the Server

```bash
uvicorn api.main:app --reload --port 8000
```
