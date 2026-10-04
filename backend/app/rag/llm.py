"""Chat model selection with fallback: each provider is tried in order until one answers."""

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)

PROVIDERS = ("gemini", "groq")


@dataclass
class ChatModel:
    name: str
    model: Any  # a LangChain chat model

    def generate(self, prompt: str) -> str:
        text = _text(self.model.invoke(prompt)).strip()
        if not text:
            raise ValueError("empty response")
        return text

    def stream(self, prompt: str) -> Iterator[str]:
        for chunk in self.model.stream(prompt):
            if piece := _text(chunk):
                yield piece


def _text(message: Any) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, list):  # some providers return content blocks
        content = "".join(block.get("text", "") if isinstance(block, dict) else str(block) for block in content)
    return str(content)


class AllProvidersFailed(RuntimeError):
    pass


@dataclass
class LLM:
    """An ordered chain of chat models; generate() falls through to the next one on any error."""

    models: list[ChatModel]

    @property
    def name(self) -> str:
        return " -> ".join(m.name for m in self.models)

    def generate(self, prompt: str) -> tuple[str, str]:
        """Return (answer, name of the model that produced it)."""
        errors = []
        for model in self.models:
            try:
                return model.generate(prompt), model.name
            except Exception as exc:
                logger.warning("%s failed, trying the next provider: %s", model.name, exc)
                errors.append(f"{model.name}: {type(exc).__name__}: {exc}")
        raise AllProvidersFailed("; ".join(errors))

    def stream(self, prompt: str) -> Iterator[tuple[str, str]]:
        """Yield (model name, text piece). A provider that fails before its first piece hands over to the
        next one; a failure after streaming has started is raised, since the output can't be taken back."""
        errors = []
        for model in self.models:
            started = False
            try:
                for piece in model.stream(prompt):
                    started = True
                    yield model.name, piece
                if started:
                    return
                raise ValueError("empty response")
            except Exception as exc:
                if started:
                    raise
                logger.warning("%s failed, trying the next provider: %s", model.name, exc)
                errors.append(f"{model.name}: {type(exc).__name__}: {exc}")
        raise AllProvidersFailed("; ".join(errors))


def _build(provider: str, settings: Settings) -> ChatModel | None:
    if provider == "gemini" and settings.gemini_api_key:
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatModel(
            f"gemini:{settings.gemini_model}",
            ChatGoogleGenerativeAI(
                google_api_key=settings.gemini_api_key,
                model=settings.gemini_model,
                temperature=settings.temperature,
                max_output_tokens=settings.max_tokens,
                timeout=settings.llm_timeout,
                max_retries=settings.llm_max_retries,
            ),
        )
    if provider == "groq" and settings.groq_api_key:
        from langchain_groq import ChatGroq

        return ChatModel(
            f"groq:{settings.groq_model}",
            ChatGroq(
                api_key=settings.groq_api_key,
                model=settings.groq_model,
                temperature=settings.temperature,
                max_tokens=settings.max_tokens,
                timeout=settings.llm_timeout,
                max_retries=settings.llm_max_retries,
            ),
        )
    return None


@lru_cache
def get_llm() -> LLM | None:
    """The provider chain from LLM_PROVIDERS, or None for retrieval-only answers."""
    settings = get_settings()
    order = [p.strip().lower() for p in settings.llm_providers.split(",") if p.strip()]
    if order in ([], ["none"]):
        return None
    unknown = [p for p in order if p not in PROVIDERS]
    if unknown:
        raise RuntimeError(f"Unknown LLM provider(s) {unknown}; choose from {list(PROVIDERS)} or 'none'")
    models = [m for m in (_build(p, settings) for p in order) if m is not None]
    if not models:
        logger.warning("No API key set for any of %s; answers will be retrieval-only", order)
        return None
    return LLM(models)
