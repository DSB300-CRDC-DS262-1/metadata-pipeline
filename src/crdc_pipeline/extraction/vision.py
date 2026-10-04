"""Page transcription from an image, for pages whose PDF text layer is unreliable."""

from __future__ import annotations

from typing import Callable

from pydantic import BaseModel, Field

from .backends import Backend, make_backend

ATTEMPTS = 2
DEFAULT_VISION_MODEL = "gemma4:e4b"
# A page image fits easily in 16K, and Gemma 4 fails to load images at 32K on a 16 GB GPU.
VISION_NUM_CTX = 16384

# Charts, figures and tables belong to the separate table/image workstream, so only prose is read.
PROMPT = (
    "Transcribe the running text on this page exactly as printed, in reading order: headings, "
    "paragraphs, quotes, captions and footnotes, including every number, percent sign and minus "
    "sign. Do not transcribe charts, graphs, infographics or tables: write [figure] in their place. "
    "Do not summarise, translate or correct anything."
)


WORD_PROMPT = (
    "This image shows one word or number from a document. Write exactly the characters you see, "
    "with no spaces, keeping symbols as printed: en dash, em dash, minus sign, equals, plus, "
    "accented letters."
)


class PageTranscription(BaseModel):
    text: str = Field(description="The page text exactly as printed.")


def make_vision_backend(kind: str = "ollama", model: str | None = None) -> Backend:
    """A backend for reading images; local models run with thinking off (faster, as accurate)."""
    if kind == "ollama":
        return make_backend("ollama", model or DEFAULT_VISION_MODEL, num_ctx=VISION_NUM_CTX, think=False)
    return make_backend(kind, model)


def make_page_transcriber(backend: Backend) -> Callable[[bytes], str]:
    def transcribe(png: bytes) -> str:
        last_error: Exception | None = None
        for _ in range(ATTEMPTS):
            try:
                text = backend.generate(PROMPT, PageTranscription, images=[png]).parsed.text
            except ValueError as exc:  # empty or malformed model output
                last_error = exc
                continue
            if text.strip():
                return text
        raise RuntimeError(f"vision transcription failed after {ATTEMPTS} attempts: {last_error}")

    return transcribe


def make_word_reader(backend: Backend) -> Callable[[bytes], str]:
    """Reads a cropped word image; an unreadable reply is treated as no answer."""

    def read(png: bytes) -> str:
        try:
            return backend.generate(WORD_PROMPT, PageTranscription, images=[png]).parsed.text
        except ValueError:
            return ""

    return read
