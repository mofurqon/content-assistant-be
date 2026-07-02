FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN chmod +x scripts/start.sh

# Bake the Chroma vector store into the image at build time. Railway's free
# tier has no persistent volumes, so chroma_db/ can't survive across
# deploys/restarts on disk — but the KB is a fixed pre-ingested PDF (see
# CLAUDE.md), not user data, so building it once per image and shipping it as
# a layer is equivalent and needs no volume at all.
# GEMINI_API_KEY must be set as a build-time variable on the Railway service
# (mark it "Sealed" in the dashboard - Railway doesn't support BuildKit
# --mount=type=secret, so this is its recommended way to keep it out of the
# UI/API; it still ends up in this layer like any ARG).
ARG GEMINI_API_KEY
ARG GEMINI_EMBED_MODEL=gemini-embedding-2
RUN GEMINI_API_KEY=$GEMINI_API_KEY GEMINI_EMBED_MODEL=$GEMINI_EMBED_MODEL python -m ingest.ingest

EXPOSE 8000
CMD ["bash", "scripts/start.sh"]
