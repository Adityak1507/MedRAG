"""Chat model selection. Providers are tried in the same order as the notebook."""

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from app.config import Settings, get_settings


@dataclass
class LLM:
    name: str
    model: Any  # a LangChain chat model

    def generate(self, prompt: str) -> str:
        reply = self.model.invoke(prompt)
        content = getattr(reply, "content", reply)
        if isinstance(content, list):  # some providers return content blocks
            content = "".join(
                block.get("text", "") if isinstance(block, dict) else str(block) for block in content
            )
        return str(content).strip()


def _build(provider: str, settings: Settings) -> LLM | None:
    if provider == "openai" and settings.openai_api_key:
        from langchain_openai import ChatOpenAI

        return LLM(
            f"openai:{settings.openai_model}",
            ChatOpenAI(
                api_key=settings.openai_api_key,
                model=settings.openai_model,
                temperature=settings.temperature,
                max_tokens=settings.max_tokens,
            ),
        )
    if provider == "anthropic" and settings.anthropic_api_key:
        from langchain_anthropic import ChatAnthropic

        # Current Claude models reject sampling parameters, so temperature is not sent
        return LLM(
            f"anthropic:{settings.anthropic_model}",
            ChatAnthropic(
                api_key=settings.anthropic_api_key,
                model=settings.anthropic_model,
                max_tokens=settings.max_tokens,
            ),
        )
    if provider == "cohere" and settings.cohere_api_key:
        from langchain_cohere import ChatCohere

        return LLM(
            f"cohere:{settings.cohere_model}",
            ChatCohere(
                cohere_api_key=settings.cohere_api_key,
                model=settings.cohere_model,
                temperature=settings.temperature,
                max_tokens=settings.max_tokens,
            ),
        )
    return None


@lru_cache
def get_llm() -> LLM | None:
    """The configured LLM, or None for retrieval-only answers."""
    settings = get_settings()
    provider = settings.llm_provider.lower()
    if provider == "none":
        return None
    if provider != "auto":
        llm = _build(provider, settings)
        if llm is None:
            raise RuntimeError(f"LLM_PROVIDER={provider} but its API key is not set")
        return llm
    for candidate in ("openai", "anthropic", "cohere"):
        llm = _build(candidate, settings)
        if llm is not None:
            return llm
    return None
