from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import src.llm_client as llm_client_module
from src.llm_client import LLMClient, LLMTimeoutError, IntentOutput, StructuredOutputError
from src.token_counter import TokenCounter


class FakeResponse:
    def __init__(self, text: str, usage=None):
        self.choices = [SimpleNamespace(message=SimpleNamespace(content=text))]
        self.usage = usage


class FakeCompletions:
    def __init__(self, response=None, error=None):
        self._response = response
        self._error = error

    def create(self, **kwargs):
        if self._error:
            raise self._error
        return self._response


class FakeClient:
    def __init__(self, response=None, error=None):
        self.chat = SimpleNamespace(
            completions=FakeCompletions(response=response, error=error)
        )


def test_missing_api_key_raises_value_error(monkeypatch):
    monkeypatch.setattr(llm_client_module, "load_project_env", lambda: None)
    monkeypatch.delenv("api_key", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="api_key is required"):
        LLMClient(api_key=None, model_name="demo-model", client=FakeClient())


def test_timeout_is_captured_as_custom_error():
    fake = FakeClient(error=TimeoutError("request timeout"))
    llm = LLMClient(api_key="k", base_url="https://example.com", model_name="m", client=fake)

    with pytest.raises(LLMTimeoutError):
        llm.chat(messages=[{"role": "user", "content": "hi"}])


def test_invalid_json_output_is_detected():
    with pytest.raises(StructuredOutputError):
        LLMClient.parse_structured_text("not-json", IntentOutput)


def test_token_counter_records_usage_fields():
    counter = TokenCounter()
    usage = counter.record("hello", "world")

    assert usage.input_tokens >= 1
    assert usage.output_tokens >= 1
    assert usage.total_tokens == usage.input_tokens + usage.output_tokens
    assert usage.cost_estimate >= 0


def test_chat_return_raw_contains_text_and_response():
    usage_obj = SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    fake_response = FakeResponse(text="ok", usage=usage_obj)
    llm = LLMClient(
        api_key="k",
        base_url="https://example.com",
        model_name="m",
        client=FakeClient(response=fake_response),
    )

    result = llm.chat(messages=[{"role": "user", "content": "hi"}], return_raw=True)
    assert result.text == "ok"
    assert result.raw_response is fake_response
    assert result.token_usage.total_tokens == 15
