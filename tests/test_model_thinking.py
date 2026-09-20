from unittest.mock import patch

import pytest
from pydantic import ValidationError

from src.core.config import LLMConfig
from src.core.model_provider import ModelProvider


@pytest.mark.parametrize(
    ("effort", "json_mode", "expected"),
    [
        ("auto", False, {}),
        ("off", False, {"thinking": {"type": "disabled"}}),
        ("low", False, {"thinking": {"type": "enabled"}, "reasoning_effort": "low"}),
        ("high", False, {"thinking": {"type": "enabled"}, "reasoning_effort": "high"}),
        ("max", False, {"thinking": {"type": "enabled"}, "reasoning_effort": "max"}),
        ("max", True, {"thinking": {"type": "disabled"}}),
    ],
)
def test_deepseek_thinking_payload(effort, json_mode, expected):
    provider = ModelProvider(LLMConfig(base_url="https://api.deepseek.com/v1", thinking_effort=effort))
    with patch("src.core.model_provider.httpx.Client") as client:
        response = client.return_value.__enter__.return_value.post.return_value
        response.json.return_value = {"choices": [{"message": {"content": " answer "}}]}
        result = provider._execute_chat_http("deepseek-flash", "system", "user", 0.7, 4096, json_mode, 10)
        payload = client.return_value.__enter__.return_value.post.call_args.kwargs["json"]

    assert result == "answer"
    assert {key: payload[key] for key in ("thinking", "reasoning_effort") if key in payload} == expected
    assert ("response_format" in payload) == json_mode


def test_thinking_controls_not_sent_to_other_compatible_providers():
    provider = ModelProvider(LLMConfig(base_url="https://proxy.example/v1", thinking_effort="max"))
    with patch("src.core.model_provider.httpx.Client") as client:
        client.return_value.__enter__.return_value.post.return_value.json.return_value = {
            "choices": [{"message": {"content": "ok"}}]
        }
        provider._execute_chat_http("other-model", "system", "user", 0.7, 4096, True, 10)
        payload = client.return_value.__enter__.return_value.post.call_args.kwargs["json"]

    assert "thinking" not in payload
    assert "reasoning_effort" not in payload
    assert payload["response_format"] == {"type": "json_object"}


def test_completion_exposes_provider_finish_reason():
    provider = ModelProvider(LLMConfig())
    with patch("src.core.model_provider.httpx.Client") as client:
        client.return_value.__enter__.return_value.post.return_value.json.return_value = {
            "choices": [{"message": {"content": "未完待续"}, "finish_reason": "length"}]
        }
        assert provider.chat_completion("system", "user") == "未完待续"

    assert provider.last_finish_reason == "length"


def test_unknown_thinking_effort_rejected():
    with pytest.raises(ValidationError):
        LLMConfig(thinking_effort="extreme")
