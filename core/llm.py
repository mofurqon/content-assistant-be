from functools import lru_cache
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from config.settings import require_env


@lru_cache(maxsize=None)
def get_llm(temperature: float = 0.5) -> ChatGoogleGenerativeAI:
    """
    Return a cached Gemini chat client for the given temperature.

    Clients are stateless across requests, so one instance per temperature is
    reused for the whole process instead of rebuilt on every call.
    """
    return ChatGoogleGenerativeAI(
        model=require_env("GEMINI_MODEL"),
        google_api_key=require_env("GEMINI_API_KEY"),
        temperature=temperature,
    )


@lru_cache(maxsize=None)
def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    """Return a cached Gemini embeddings client."""
    return GoogleGenerativeAIEmbeddings(
        model=f"models/{require_env('GEMINI_EMBED_MODEL')}",
        google_api_key=require_env("GEMINI_API_KEY"),
    )
