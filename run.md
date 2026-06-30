# /run

Start the Streamlit UI.

## What it does
Launches the AI Content Assistant in the browser. All agent steps run from here.

## Prerequisites
- Ingest pipeline must have been run first (`/ingest`)
- GEMINI_API_KEY must be set in .env

## Command
```bash
streamlit run app.py
```

## Expected output
- Browser opens at http://localhost:8501
- User fills in: Topic, Target Audience, Content Type, Tone
- Agent runs step by step with streaming output
- User selects idea from generated list
- Agent continues through all remaining steps
