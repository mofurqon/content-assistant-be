# Threshold Calibration Script

One-time offline script to find the optimal cosine similarity threshold for
Gemini `gemini-embedding-2`, replacing the hardcoded `0.7` in the RAG retrieval node.

---

## Context

| Item | Value |
|---|---|
| Embedding model | Gemini `gemini-embedding-2` |
| Vector store | Chroma (local) |
| Knowledge base | `software-testing-guide-book.pdf` (already ingested) |
| Current threshold | `0.7` (hardcoded in RAG retrieval node) |
| Framework | Python, FastAPI, LangGraph |

---

## Output Location

```
scripts/calibrate_threshold.py
scripts/calibration_result.json   ← generated after running
```

---

## Steps

### Step 1 — Load existing chunks from Chroma

Read all existing embedded chunks directly from the Chroma collection.

- Do **NOT** re-ingest the PDF
- Reuse the existing Chroma client initialization from `backend/services/` or `backend/core/`
- Retrieve chunks via:

```python
collection.get(include=["embeddings", "documents"])
```

---

### Step 2 — Synthetic QA generation

For each chunk (or a random sample of max **80 chunks** if the collection is large),
call Gemini Flash LLM to generate 3 questions directly answered by that chunk.

Use this exact prompt:

```
Generate exactly 3 short questions that are directly answered by the text below.
Return only the questions, one per line, no numbering, no extra text.

Text: {chunk}
```

Each (question, source_chunk) pair is a **positive pair** → label = `1`

---

### Step 3 — Generate negative pairs

For each question, randomly sample **3 chunks** that are NOT the source chunk.

Each (question, random_chunk) pair is a **negative pair** → label = `0`

Target ratio: ~1:3 positive to negative (realistic imbalance).

---

### Step 4 — Compute cosine similarity scores

For each (question, chunk) pair:

1. Embed the question using Gemini `gemini-embedding-2` (same model as the pipeline)
2. Retrieve the chunk's existing embedding from Chroma (do **not** re-embed)
3. Compute cosine similarity:

```python
import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
```

> **Important**: The calibration script must use the same embedding model that was used
> during ingest. If you switch models, re-ingest the PDF first, then re-run calibration.

---

### Step 5 — Precision-Recall curve and best threshold

Use `sklearn.metrics.precision_recall_curve` to sweep thresholds.
Compute F1 at each threshold and find the one with max F1.

Print a summary table:

```
Threshold | Precision | Recall | F1
----------|-----------|--------|----
0.40      | 0.61      | 0.92   | 0.73
0.46      | 0.74      | 0.85   | 0.79  ← best F1
0.52      | 0.83      | 0.71   | 0.77
...

✅ Recommended threshold: 0.46
   Precision: 0.74 | Recall: 0.85 | F1: 0.79

   Update RAG_SIMILARITY_THRESHOLD in your .env
```

---

### Step 6 — Save result

Save full results to `scripts/calibration_result.json`:

```json
{
  "recommended_threshold": 0.46,
  "f1": 0.79,
  "precision": 0.74,
  "recall": 0.85,
  "model": "gemini-embedding-2",
  "pairs_evaluated": 240,
  "generated_at": "2026-06-28T10:00:00"
}
```

---

## Implementation Notes

- Load env vars from `.env` — requires `GEMINI_API_KEY` and Chroma collection name/path
- Add a **0.5s sleep** between Gemini embedding API calls to avoid free-tier rate limiting
- If total chunks > 80, use `random.sample()` to cap at 80 chunks for speed
- Do **NOT** modify any existing pipeline code

---

## Dependencies

Add to `requirements.txt` if missing:

```
scikit-learn
numpy
matplotlib   # optional, for plotting the curve
```

---

## How to Run

```bash
python scripts/calibrate_threshold.py
```

After running, update your `.env`:

```env
RAG_SIMILARITY_THRESHOLD=0.46   # replace with your actual result
```

Then read it in the RAG retrieval node via `os.getenv()` instead of hardcoding.

---

## Notes

- **Rate limits** — Gemini free tier has tight RPM limits. If you hit errors, reduce
  the sample size to 50–60 chunks.
- **Result is a starting point** — synthetic QA pairs are LLM-generated so there is
  inherent noise. Use the result as a calibrated baseline, not a precision instrument.
- **Re-run when** you switch embedding models, add a significantly different corpus
  domain, or notice retrieval quality degrading in production.
