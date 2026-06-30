from functools import lru_cache
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from config.settings import require_env
from core.log import LLMLoggingHandler

_llm_handler = LLMLoggingHandler()


@lru_cache(maxsize=None)
def get_llm(temperature: float = 0.5) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=require_env("GEMINI_MODEL"),
        google_api_key=require_env("GEMINI_API_KEY"),
        temperature=temperature,
        callbacks=[_llm_handler],
    )


@lru_cache(maxsize=None)
def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    """Return a cached Gemini embeddings client."""
    return GoogleGenerativeAIEmbeddings(
        model=f"models/{require_env('GEMINI_EMBED_MODEL')}",
        google_api_key=require_env("GEMINI_API_KEY"),
    )
