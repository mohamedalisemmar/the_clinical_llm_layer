"""LLM client abstraction.

Every call goes through `LLMClient.complete(system, user, task=..., meta=...)`.
Real providers ignore `task` and `meta`; the mock provider uses them to
simulate plausible behaviour offline.
"""
from __future__ import annotations

import os
import time
from typing import Any, Optional

import config


class LLMError(RuntimeError):
    pass


class LLMClient:
    provider = "base"

    def __init__(self, model: str):
        self.model = model
        self.n_calls = 0
        # Usage accounting (for the "is the layer lightweight?" analysis).
        self.latency_s = 0.0
        self.input_tokens = 0
        self.output_tokens = 0
        self.calls_by_task: dict[str, int] = {}
        self.latency_by_task: dict[str, float] = {}
        self.tokens_by_task: dict[str, list[int]] = {}

    def usage_snapshot(self) -> dict:
        return {"calls": self.n_calls, "latency_s": self.latency_s, "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens, "calls_by_task": dict(self.calls_by_task),
                "latency_by_task": dict(self.latency_by_task),
                "tokens_by_task": {k: list(v) for k, v in self.tokens_by_task.items()}}

    def _record(self, task: str, seconds: float, tokens_in: int, tokens_out: int) -> None:
        self.latency_s += seconds
        self.input_tokens += tokens_in
        self.output_tokens += tokens_out
        self.calls_by_task[task] = self.calls_by_task.get(task, 0) + 1
        self.latency_by_task[task] = self.latency_by_task.get(task, 0.0) + seconds
        tok = self.tokens_by_task.setdefault(task, [0, 0])
        tok[0] += tokens_in
        tok[1] += tokens_out

    def complete(
        self,
        system: str,
        user: str,
        *,
        task: str,
        meta: Optional[dict[str, Any]] = None,
        temperature: float = 0.7,
        max_tokens: int = 1500,
    ) -> str:
        last_err: Exception | None = None
        for attempt in range(4):
            try:
                self.n_calls += 1
                self._last_usage = None
                t0 = time.perf_counter()
                text = self._complete(system, user, task=task, meta=meta or {},
                                      temperature=temperature, max_tokens=max_tokens)
                usage = self._last_usage or (_estimate_tokens(system + user), _estimate_tokens(text))
                self._record(task, time.perf_counter() - t0, *usage)
                return text
            except Exception as err:  # network / rate-limit errors
                last_err = err
                time.sleep(2 ** attempt)
        raise LLMError(f"{self.provider} call failed after retries: {last_err}")

    def _complete(self, system, user, *, task, meta, temperature, max_tokens) -> str:
        raise NotImplementedError


class AnthropicClient(LLMClient):
    provider = "anthropic"

    def __init__(self, model: str):
        super().__init__(model)
        import anthropic  # imported lazily so the package is optional
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise LLMError("ANTHROPIC_API_KEY is not set (see .env.example)")
        self._client = anthropic.Anthropic()

    def _complete(self, system, user, *, task, meta, temperature, max_tokens) -> str:
        resp = self._client.messages.create(
            model=self.model,
            system=system,
            messages=[{"role": "user", "content": user}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if getattr(resp, "usage", None):
            self._last_usage = (resp.usage.input_tokens, resp.usage.output_tokens)
        return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")


class OpenAICompatClient(LLMClient):
    """OpenAI, Groq, OpenRouter, Ollama, LM Studio... anything OpenAI-compatible."""
    provider = "openai"

    def __init__(self, model: str):
        super().__init__(model)
        from openai import OpenAI
        api_key = os.getenv("OPENAI_API_KEY") or ("ollama" if config.OPENAI_BASE_URL else None)
        if not api_key:
            raise LLMError("OPENAI_API_KEY is not set (see .env.example)")
        self._client = OpenAI(api_key=api_key, base_url=config.OPENAI_BASE_URL)

    def _complete(self, system, user, *, task, meta, temperature, max_tokens) -> str:
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if getattr(resp, "usage", None):
            self._last_usage = (resp.usage.prompt_tokens or 0, resp.usage.completion_tokens or 0)
        return resp.choices[0].message.content or ""


def _estimate_tokens(text: str) -> int:
    """Rough estimate (about 4 characters per token) when the API does not report usage."""
    return max(1, len(text or "") // 4)


def get_client(provider: Optional[str] = None, model: Optional[str] = None) -> LLMClient:
    provider = (provider or config.LLM_PROVIDER).lower()
    model = model or config.LLM_MODEL or config.DEFAULT_MODELS.get(provider, "")
    if provider == "mock":
        from llm.mock import MockClient
        return MockClient(model or "mock-clinical-simulator")
    if provider == "anthropic":
        return AnthropicClient(model)
    if provider == "openai":
        return OpenAICompatClient(model)
    raise LLMError(f"Unknown LLM_PROVIDER '{provider}' (use mock, anthropic or openai)")
