from __future__ import annotations
from pydantic import BaseModel


class Idea(BaseModel):
    text: str


class KBChunk(BaseModel):
    content: str
    score: float | None = None


class EvalResult(BaseModel):
    scores: dict[str, int]
    average: float
    reasoning: str


class ResearchResult(BaseModel):
    queries: list[str]
    summary: str


class PipelineResult(BaseModel):
    idea: str
    kb_chunks: list[KBChunk]
    draft: str
    evaluations: list[EvalResult]
    research: ResearchResult
    article: str
    image_prompt: str
