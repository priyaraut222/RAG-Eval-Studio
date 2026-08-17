"""
A thin, provider-agnostic LLM client.

Every place in the app that needs to call an LLM (synthetic dataset
generation now, faithfulness/hallucination judging in Phase 4) goes
through `LLMClient.complete()` rather than importing `openai` or
`google.generativeai` directly. That keeps provider-switching a
one-line config change and makes the call sites trivially mockable
in tests.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Protocol

from backend.config.settings import Settings, get_settings
from backend.utils.logger import get_logger

logger = get_logger(__name__)


class LLMError(Exception):
    """Raised when an LLM call fails or is misconfigured (e.g. missing API key)."""


@dataclass
class LLMResponse:
    """Normalized result of an LLM call, regardless of provider."""

    text: str
    provider: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class _Provider(Protocol):
    def complete(self, prompt: str, system: str | None, temperature: float) -> LLMResponse: ...


class _OpenAIProvider:
    def __init__(self, api_key: str, model: str) -> None:
        self._api_key = api_key
        self._model = model

    def complete(self, prompt: str, system: str | None, temperature: float) -> LLMResponse:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise LLMError("openai package is not installed.") from exc

        client = OpenAI(api_key=self._api_key)
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        try:
            response = client.chat.completions.create(
                model=self._model, messages=messages, temperature=temperature
            )
        except Exception as exc:
            raise LLMError(f"OpenAI request failed: {exc}") from exc

        choice = response.choices[0].message.content or ""
        usage = response.usage
        return LLMResponse(
            text=choice,
            provider="openai",
            model=self._model,
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
        )


class _GeminiProvider:
    def __init__(self, api_key: str, model: str) -> None:
        self._api_key = api_key
        self._model = model

    def complete(self, prompt: str, system: str | None, temperature: float) -> LLMResponse:
        try:
            import google.generativeai as genai
        except ImportError as exc:  # pragma: no cover
            raise LLMError("google-generativeai package is not installed.") from exc

        genai.configure(api_key=self._api_key)
        model = genai.GenerativeModel(
            model_name=self._model,
            system_instruction=system,
        )
        try:
            response = model.generate_content(
                prompt, generation_config={"temperature": temperature}
            )
        except Exception as exc:
            raise LLMError(f"Gemini request failed: {exc}") from exc

        usage = getattr(response, "usage_metadata", None)
        return LLMResponse(
            text=response.text or "",
            provider="gemini",
            model=self._model,
            input_tokens=getattr(usage, "prompt_token_count", 0) or 0,
            output_tokens=getattr(usage, "candidates_token_count", 0) or 0,
        )


class _LocalHeuristicProvider:
    """Deterministic, dependency-free stand-in used when no API key is configured.

    This is NOT a language model — it's a rule-based generator that lets
    the Dataset Builder (and, from Phase 4, evaluation) run end-to-end
    with zero setup so a reviewer can try the app without any API key.
    It extracts a plausible question/ground-truth pair directly from the
    source passage embedded in the synthesizer's prompt. Real usage
    should configure `openai` or `gemini` in Settings for actual quality.
    """

    _SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
    _PASSAGE_RE = re.compile(r'"""\s*(.*?)\s*"""', re.DOTALL)
    _COUNT_RE = re.compile(r"exactly (\d+) question")
    _STOPWORDS = {"this", "that", "with", "from", "have", "which", "their", "there", "these", "those"}

    def complete(self, prompt: str, system: str | None, temperature: float) -> LLMResponse:
        passage_match = self._PASSAGE_RE.search(prompt)
        passage = passage_match.group(1) if passage_match else prompt

        sentences = [s.strip() for s in self._SENTENCE_SPLIT_RE.split(passage) if s.strip()]
        count_match = self._COUNT_RE.search(prompt)
        n = int(count_match.group(1)) if count_match else 1
        n = max(1, min(n, len(sentences) or 1))

        records = []
        for i in range(n):
            sentence = sentences[i % len(sentences)] if sentences else passage.strip()[:200]
            topic = self._guess_topic(sentence)
            records.append(
                {
                    "question": f"What does the passage say about {topic}?",
                    "ground_truth": sentence,
                    "expected_chunk": sentence,
                }
            )

        return LLMResponse(
            text=json.dumps(records),
            provider="local",
            model="heuristic-v1",
            input_tokens=len(prompt) // 4,
            output_tokens=sum(len(r["ground_truth"]) for r in records) // 4,
        )

    @classmethod
    def _guess_topic(cls, sentence: str) -> str:
        words = re.findall(r"[A-Za-z][A-Za-z\-]{3,}", sentence)
        candidates = [w for w in words if w.lower() not in cls._STOPWORDS]
        if not candidates:
            return "this topic"
        for w in candidates:
            if w[0].isupper() and candidates.index(w) != 0:
                return w
        return candidates[0].lower()


class LLMClient:
    """Facade that picks the configured provider and exposes one `complete()` call.

    Falls back to the local heuristic provider whenever the selected
    cloud provider has no API key configured, rather than raising —
    `self.active_provider` tells the caller (the Streamlit UI) whether
    that fallback happened, so it can surface a friendly notice instead
    of silently using lower-quality output.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self.active_provider: str = self._settings.default_llm_provider
        self.model_name: str = "heuristic-v1"
        self._provider = self._build_provider()

    def _build_provider(self) -> _Provider:
        provider = self._settings.default_llm_provider

        if provider == "openai":
            if self._settings.openai_api_key:
                self.model_name = self._settings.default_openai_model
                return _OpenAIProvider(self._settings.openai_api_key, self._settings.default_openai_model)
            logger.warning("OpenAI selected but OPENAI_API_KEY is not set — falling back to local heuristic provider")
            self.active_provider = "local"
            return _LocalHeuristicProvider()

        if provider == "gemini":
            if self._settings.google_api_key:
                self.model_name = self._settings.default_gemini_model
                return _GeminiProvider(self._settings.google_api_key, self._settings.default_gemini_model)
            logger.warning("Gemini selected but GOOGLE_API_KEY is not set — falling back to local heuristic provider")
            self.active_provider = "local"
            return _LocalHeuristicProvider()

        self.active_provider = "local"
        return _LocalHeuristicProvider()

    def complete(self, prompt: str, system: str | None = None, temperature: float = 0.3) -> LLMResponse:
        logger.debug(f"LLM call via {self.active_provider} ({len(prompt)} char prompt)")
        return self._provider.complete(prompt, system, temperature)


def get_llm_client(settings: Settings | None = None) -> LLMClient:
    """Convenience factory — builds an `LLMClient` from current settings."""
    return LLMClient(settings=settings)
