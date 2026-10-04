"""Backend tests. The Ollama HTTP layer is mocked so no server is needed."""

import io
import json
import urllib.error

import pytest

from crdc_pipeline.extraction import DocumentMetadata, OllamaBackend, extract_metadata, make_backend


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def ollama_body(content: dict, prompt_tokens: int = 1200) -> dict:
    return {
        "model": "gemma3:12b",
        "message": {"role": "assistant", "content": json.dumps(content)},
        "prompt_eval_count": prompt_tokens,
        "eval_count": 80,
        "total_duration": 4_500_000_000,
    }


def patch_urlopen(monkeypatch, body: dict, captured: dict | None = None):
    def fake_urlopen(request, timeout=None):
        if captured is not None:
            captured["url"] = request.full_url
            captured["payload"] = json.loads(request.data)
        return FakeResponse(json.dumps(body).encode())

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)


def test_ollama_backend_parses_structured_output(monkeypatch) -> None:
    captured: dict = {}
    body = ollama_body({"title": "A Title", "authors": ["A. Author"], "organisation": "A Uni"})
    patch_urlopen(monkeypatch, body, captured)

    result = extract_metadata(["page one"], OllamaBackend("gemma3:12b"))

    assert result.metadata.title == "A Title"
    assert result.metadata.authors == ["A. Author"]
    assert result.metadata.publisher is None
    assert result.resolved_model == "gemma3:12b"
    assert result.seconds == pytest.approx(4.5)
    assert (result.prompt_tokens, result.output_tokens) == (1200, 80)


def test_ollama_request_sets_schema_and_context_window(monkeypatch) -> None:
    captured: dict = {}
    patch_urlopen(monkeypatch, ollama_body({"title": "T"}), captured)

    extract_metadata(["page one"], OllamaBackend("gemma3:12b", num_ctx=8192))

    payload = captured["payload"]
    assert captured["url"].endswith("/api/chat")
    assert payload["stream"] is False
    assert payload["options"] == {"temperature": 0, "num_ctx": 8192}
    properties = DocumentMetadata.model_json_schema()["properties"]
    assert payload["format"]["properties"].keys() == properties.keys()
    assert payload["format"]["required"] == list(properties)


def test_ollama_backend_rejects_truncated_prompt(monkeypatch) -> None:
    patch_urlopen(monkeypatch, ollama_body({"title": "T"}, prompt_tokens=4096))

    with pytest.raises(RuntimeError, match="truncated"):
        extract_metadata(["page one"], OllamaBackend("gemma3:12b", num_ctx=4096))


def test_ollama_backend_reports_unreachable_server(monkeypatch) -> None:
    def refuse(request, timeout=None):
        raise urllib.error.URLError("refused")

    monkeypatch.setattr("urllib.request.urlopen", refuse)

    with pytest.raises(RuntimeError, match="Is it running"):
        extract_metadata(["page one"], OllamaBackend("gemma3:12b"))


def test_make_backend_validates_arguments() -> None:
    assert make_backend("ollama", "llama3.1:8b").label == "ollama_llama3.1:8b"
    assert make_backend().label == "ollama_gemma4:e4b"
    with pytest.raises(ValueError, match="Unknown backend"):
        make_backend("nope")


def test_ollama_request_sends_images_base64(monkeypatch) -> None:
    import base64

    from crdc_pipeline.extraction.vision import make_page_transcriber

    captured: dict = {}
    patch_urlopen(monkeypatch, ollama_body({"text": "73% agreed"}), captured)

    text = make_page_transcriber(OllamaBackend("gemma4:12b"))(b"png-bytes")

    assert text == "73% agreed"
    images = captured["payload"]["messages"][0]["images"]
    assert images == [base64.b64encode(b"png-bytes").decode("ascii")]


def test_page_transcriber_retries_empty_output() -> None:
    from crdc_pipeline.extraction.backends import GenerationResult
    from crdc_pipeline.extraction.vision import PageTranscription, make_page_transcriber

    replies = iter(["", "73% agreed"])

    class FakeBackend:
        label = "fake"

        def generate(self, prompt, schema, images=None):
            return GenerationResult(PageTranscription(text=next(replies)), "fake", 0.0, None, None)

    assert make_page_transcriber(FakeBackend())(b"png") == "73% agreed"


def test_ollama_retries_once_when_the_answer_is_cut_off(monkeypatch) -> None:
    replies = iter([
        dict(ollama_body({"title": "T"}), done_reason="length", message={"role": "assistant", "content": '{"title'}),
        ollama_body({"title": "T"}),
    ])

    def fake_urlopen(request, timeout=None):
        return FakeResponse(json.dumps(next(replies)).encode())

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    result = extract_metadata(["page one"], OllamaBackend("gemma4:12b"))
    assert result.metadata.title == "T"


def test_schema_caps_list_lengths() -> None:
    properties = DocumentMetadata.model_json_schema()["properties"]
    assert properties["evidence"]["maxItems"] == 25
    assert properties["authors"]["maxItems"] == 50
