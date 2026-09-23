"""Model backends for structured (JSON-schema) extraction."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Protocol

from pydantic import BaseModel

MAX_RETRIES = 4
RETRY_BACKOFF_SECONDS = 5

DEFAULT_GEMINI_MODEL = "gemini-3.6-flash"

DEFAULT_OLLAMA_HOST = "http://localhost:11434"

# Ollama silently truncates prompts beyond num_ctx (default 4096). Measured
# real prompts on the 4 sample documents: metadata stays under 7K tokens,
# but statistics (whole-document prompts) hit 19K-22K on Hedayati/VoconiQ,
# plus room for a long JSON response, so 16384 isn't enough.
DEFAULT_NUM_CTX = 32768


@dataclass
class GenerationResult:
    parsed: BaseModel
    model: str
    seconds: float
    prompt_tokens: int | None
    output_tokens: int | None


class Backend(Protocol):
    label: str

    def generate(self, prompt: str, schema: type[BaseModel]) -> GenerationResult: ...


class GeminiBackend:
    def __init__(self, model: str = DEFAULT_GEMINI_MODEL) -> None:
        if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
            raise RuntimeError(
                "No GEMINI_API_KEY (or GOOGLE_API_KEY) found in the environment. "
                "Copy .env.example to .env and fill in your key, or export it directly."
            )
        from google import genai

        self.client = genai.Client()
        self.model = model
        self.label = f"gemini_{model}"

    def generate(self, prompt: str, schema: type[BaseModel]) -> GenerationResult:
        from google.genai import errors, types

        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=schema,
            temperature=0,
        )
        start = time.monotonic()
        last_error: Exception | None = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = self.client.models.generate_content(
                    model=self.model, contents=prompt, config=config
                )
                break
            except errors.ServerError as exc:
                last_error = exc
                if attempt == MAX_RETRIES:
                    raise last_error
                time.sleep(RETRY_BACKOFF_SECONDS * attempt)
        seconds = time.monotonic() - start

        parsed = response.parsed
        if parsed is None:
            parsed = schema.model_validate(json.loads(response.text))
        usage = response.usage_metadata
        return GenerationResult(
            parsed=parsed,
            model=getattr(response, "model_version", None) or self.model,
            seconds=seconds,
            prompt_tokens=getattr(usage, "prompt_token_count", None),
            output_tokens=getattr(usage, "candidates_token_count", None),
        )


class OllamaBackend:
    def __init__(
        self,
        model: str,
        host: str = DEFAULT_OLLAMA_HOST,
        num_ctx: int = DEFAULT_NUM_CTX,
        timeout: float = 900,
    ) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.num_ctx = num_ctx
        self.timeout = timeout
        self.label = f"ollama_{model}"

    def generate(self, prompt: str, schema: type[BaseModel]) -> GenerationResult:
        json_schema = schema.model_json_schema()
        # With every field optional, an empty object is a valid answer.
        json_schema["required"] = list(json_schema["properties"])
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "format": json_schema,
            "options": {"temperature": 0, "num_ctx": self.num_ctx},
        }
        request = urllib.request.Request(
            f"{self.host}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.load(response)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Ollama returned HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Could not reach Ollama at {self.host}. Is it running?"
            ) from exc

        prompt_tokens = body.get("prompt_eval_count")
        if prompt_tokens is not None and prompt_tokens >= self.num_ctx - 8:
            raise RuntimeError(
                f"Prompt ({prompt_tokens} tokens) filled the {self.num_ctx}-token context "
                "window and was likely truncated; raise num_ctx."
            )

        parsed = schema.model_validate_json(body["message"]["content"])
        return GenerationResult(
            parsed=parsed,
            model=body.get("model", self.model),
            seconds=body.get("total_duration", 0) / 1e9,
            prompt_tokens=prompt_tokens,
            output_tokens=body.get("eval_count"),
        )


def make_backend(kind: str, model: str | None = None, **kwargs) -> Backend:
    if kind == "gemini":
        return GeminiBackend(model or DEFAULT_GEMINI_MODEL)
    if kind == "ollama":
        if not model:
            raise ValueError("An Ollama model name is required, e.g. gemma3:12b")
        return OllamaBackend(model, **kwargs)
    raise ValueError(f"Unknown backend {kind!r} (expected 'gemini' or 'ollama')")
