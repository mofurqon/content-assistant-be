import logging
import os
import time

from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.outputs import LLMResult


def setup_logging() -> None:
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=getattr(logging, level, logging.INFO),
    )


class LLMLoggingHandler(AsyncCallbackHandler):
    def __init__(self) -> None:
        self._logger = logging.getLogger("llm")
        self._start_times: dict[str, float] = {}

    async def on_llm_start(self, serialized: dict, prompts: list[str], **kwargs) -> None:
        run_id = str(kwargs.get("run_id", ""))
        model = serialized.get("kwargs", {}).get("model", "unknown")
        preview = prompts[0][:200].replace("\n", " ") if prompts else ""
        self._start_times[run_id] = time.monotonic()
        self._logger.info("START model=%s | prompt_preview=%r", model, preview)

    async def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        run_id = str(kwargs.get("run_id", ""))
        elapsed = time.monotonic() - self._start_times.pop(run_id, time.monotonic())
        usage = (response.llm_output or {}).get("usage_metadata", {})
        self._logger.info(
            "END latency=%.2fs | tokens_in=%s tokens_out=%s",
            elapsed,
            usage.get("input_tokens", "?"),
            usage.get("output_tokens", "?"),
        )

    async def on_llm_error(self, error: BaseException, **kwargs) -> None:
        self._logger.error("ERROR %s: %s", type(error).__name__, error)
