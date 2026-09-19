from unittest.mock import patch

import httpx
import pytest

from src.core.config import EmbeddingConfig, LLMConfig
from src.core.exceptions import EmbeddingUnavailableError
from src.core.model_provider import ModelProvider


def _provider(**embedding_overrides) -> ModelProvider:
    config = EmbeddingConfig(
        api_key="test-key",
        base_url="https://embedding.example/v1",
        model="test-embedding",
        retry_base_delay=0,
        **embedding_overrides,
    )
    return ModelProvider(LLMConfig(), config)


def _response(status: int, *, data=None, headers=None) -> httpx.Response:
    request = httpx.Request("POST", "https://embedding.example/v1/embeddings")
    payload = data if data is not None else {"error": "temporary failure"}
    return httpx.Response(status, request=request, json=payload, headers=headers)


def test_embeddings_are_requested_in_bounded_batches_and_keep_input_order():
    provider = _provider(batch_size=2, max_retries=0)
    vector_by_text = {
        "a": [1.0, 0.0],
        "b": [2.0, 0.0],
        "c": [3.0, 0.0],
        "d": [4.0, 0.0],
        "e": [5.0, 0.0],
    }

    def post(_url, *, headers, json):
        items = [
            {"index": index, "embedding": vector_by_text[text]}
            for index, text in enumerate(json["input"])
        ]
        return _response(200, data={"data": list(reversed(items))})

    with patch("src.core.model_provider.httpx.Client") as client_cls:
        client = client_cls.return_value.__enter__.return_value
        client.post.side_effect = post
        result = provider.get_embeddings(["a", "b", "c", "d", "e"])

    assert result == [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0], [5.0, 0.0]]
    assert client.post.call_count == 3
    assert [call.kwargs["json"]["input"] for call in client.post.call_args_list] == [
        ["a", "b"],
        ["c", "d"],
        ["e"],
    ]


def test_embedding_retries_transient_statuses_then_succeeds():
    provider = _provider(batch_size=8, max_retries=2)
    responses = [
        _response(429, headers={"Retry-After": "0"}),
        _response(503),
        _response(200, data={"data": [{"index": 0, "embedding": [0.1, 0.2]}]}),
    ]

    with patch("src.core.model_provider.httpx.Client") as client_cls, patch(
        "src.core.model_provider.time.sleep"
    ) as sleep:
        client = client_cls.return_value.__enter__.return_value
        client.post.side_effect = responses
        result = provider.get_embeddings(["测试文本"])

    assert result == [[0.1, 0.2]]
    assert client.post.call_count == 3
    assert sleep.call_count == 2


def test_embedding_provider_error_is_not_silently_converted_to_none():
    provider = _provider(max_retries=0)

    with patch("src.core.model_provider.httpx.Client") as client_cls:
        client = client_cls.return_value.__enter__.return_value
        client.post.return_value = _response(401, data={"message": "invalid token"})
        with pytest.raises(EmbeddingUnavailableError, match="HTTP 401"):
            provider.get_embeddings(["测试文本"])


def test_embedding_malformed_response_fails_closed():
    provider = _provider(max_retries=0)

    with patch("src.core.model_provider.httpx.Client") as client_cls:
        client = client_cls.return_value.__enter__.return_value
        client.post.return_value = _response(200, data={"data": []})
        with pytest.raises(EmbeddingUnavailableError, match="返回向量数量不匹配"):
            provider.get_embeddings(["测试文本"])


def test_missing_embedding_credentials_keeps_explicit_sparse_only_mode():
    provider = ModelProvider(
        LLMConfig(api_key=""),
        EmbeddingConfig(api_key="", base_url="", model="test-embedding"),
    )

    with patch("src.core.model_provider.httpx.Client") as client_cls:
        assert provider.get_embeddings(["测试文本"]) is None
        client_cls.assert_not_called()
    assert not provider.has_embedding_credentials()
