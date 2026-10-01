"""Chat model selection with fallback: each provider is tried in order until one answers."""

import logging
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
        reply = self.model.invoke(prompt)
        content = getattr(reply, "content", reply)
        if isinstance(content, list):  # some providers return content blocks
            content = "".join(
                block.get("text", "") if isinstance(block, dict) else str(block) for block in content
            )
        text = str(content).strip()
        if not text:
            raise ValueError("empty response")
        return text


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
