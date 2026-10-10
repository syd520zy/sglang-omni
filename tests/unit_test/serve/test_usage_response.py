# SPDX-License-Identifier: Apache-2.0
"""Response usage serialization and streaming opt-in contract."""

import asyncio
import json

import pytest
from fastapi.testclient import TestClient

from sglang_omni.client.types import (
    CompletionResult,
    CompletionStreamChunk,
    GenerateRequest,
    UsageInfo,
)
from sglang_omni.serve.openai_api import (
    _build_generate_response as build_generate_response,
)
from sglang_omni.serve.openai_api import _chat_stream as chat_stream
from sglang_omni.serve.openai_api import create_app
from sglang_omni.serve.protocol import (
    ChatCompletionRequest,
    RolloutGenerateRequest,
    UsageResponse,
)


def test_usage_response_omits_unavailable_details():
    assert UsageResponse().model_dump() == {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
    data = UsageResponse(
        prompt_tokens_details={"cached_tokens": 0}, reasoning_tokens=0
    ).model_dump()
    assert data["prompt_tokens_details"] == {"cached_tokens": 0}
    assert data["reasoning_tokens"] == 0


def test_generate_returns_actual_cache_count():
    response = build_generate_response(
        RolloutGenerateRequest(prompt="hello", return_logprob=False),
        CompletionResult(
            request_id="cache",
            text="world",
            usage=UsageInfo(prompt_tokens_details={"cached_tokens": 7}),
        ),
        "wav",
    )
    assert response.meta_info.cached_tokens == 7


@pytest.mark.parametrize("include_usage", [False, True])
def test_stream_usage_option_and_terminal_order(include_usage):
    class StreamClient:
        async def completion_stream(self, *args, **kwargs):
            yield CompletionStreamChunk(
                request_id="stream",
                text="hello",
                usage=UsageInfo(prompt_tokens=3, completion_tokens=1, total_tokens=4),
            )
            yield CompletionStreamChunk(request_id="stream", finish_reason="stop")

    async def collect():
        return [
            event
            async for event in chat_stream(
                client=StreamClient(),
                gen_req=GenerateRequest(prompt="hello", stream=True),
                request_id="stream",
                response_id="chatcmpl-stream",
                created=0,
                model="mock",
                req=ChatCompletionRequest(
                    messages=[{"role": "user", "content": "hello"}],
                    stream=True,
                    stream_options={"include_usage": include_usage},
                ),
                audio_format="wav",
            )
        ]

    events = asyncio.run(collect())
    assert events[-1] == "data: [DONE]\n\n"
    chunks = [json.loads(event[6:]) for event in events[:-1]]
    if include_usage:
        assert chunks[-1]["choices"] == []
        assert chunks[-1]["usage"]["total_tokens"] == 4
        assert chunks[-2]["choices"][0]["finish_reason"] == "stop"
    else:
        assert all("usage" not in chunk for chunk in chunks)
        assert chunks[-1]["choices"][0]["finish_reason"] == "stop"


@pytest.mark.parametrize(
    "usage",
    [
        None,
        UsageInfo(
            prompt_tokens=12,
            completion_tokens=3,
            total_tokens=15,
            prompt_tokens_details={"cached_tokens": 0, "audio_tokens": 8},
            reasoning_tokens=0,
        ),
    ],
)
def test_nonstreaming_chat_usage_details(usage):
    class ResultClient:
        def health(self):
            return {"running": True}

        async def completion(self, request, *, request_id, audio_format):
            return CompletionResult(request_id=request_id, text="hello", usage=usage)

    with TestClient(create_app(ResultClient(), model_name="mock")) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "hello"}],
            },
        )
    assert response.status_code == 200
    body = response.json()
    assert body["choices"][0]["message"]["content"] == "hello"
    if usage is None:
        assert body["usage"] is None
    else:
        assert body["usage"]["prompt_tokens_details"] == {
            "cached_tokens": 0,
            "audio_tokens": 8,
        }
        assert body["usage"]["reasoning_tokens"] == 0


@pytest.mark.parametrize("fail", [False, True])
def test_missing_usage_or_failure_does_not_manufacture_statistics(fail):
    class StreamClient:
        async def completion_stream(self, *args, **kwargs):
            yield CompletionStreamChunk(request_id="missing", text="hello")
            if fail:
                raise RuntimeError("backend failed")
            yield CompletionStreamChunk(request_id="missing", finish_reason="stop")

    async def collect():
        return [
            event
            async for event in chat_stream(
                client=StreamClient(),
                gen_req=GenerateRequest(prompt="hello", stream=True),
                request_id="missing",
                response_id="chatcmpl-missing",
                created=0,
                model="mock",
                req=ChatCompletionRequest(
                    messages=[{"role": "user", "content": "hello"}],
                    stream=True,
                    stream_options={"include_usage": True},
                ),
                audio_format="wav",
            )
        ]

    if fail:
        with pytest.raises(RuntimeError, match="backend failed"):
            asyncio.run(collect())
        return
    events = asyncio.run(collect())
    assert events[-1] == "data: [DONE]\n\n"
    final = json.loads(events[-2][6:])
    if fail:
        assert "error" in final
        assert not any('"usage"' in event for event in events)
    else:
        assert final["choices"] == []
        assert final["usage"] is None


def test_closing_chat_stream_closes_backend_without_final_usage():
    closed = []

    class StreamClient:
        async def completion_stream(self, *args, **kwargs):
            try:
                yield CompletionStreamChunk(
                    request_id="cancel", text="hello", usage=UsageInfo(total_tokens=1)
                )
                yield CompletionStreamChunk(request_id="cancel", finish_reason="stop")
            finally:
                closed.append(True)

    async def cancel():
        stream = chat_stream(
            client=StreamClient(),
            gen_req=GenerateRequest(prompt="hello", stream=True),
            request_id="cancel",
            response_id="chatcmpl-cancel",
            created=0,
            model="mock",
            req=ChatCompletionRequest(
                messages=[{"role": "user", "content": "hello"}],
                stream=True,
                stream_options={"include_usage": True},
            ),
            audio_format="wav",
        )
        await anext(stream)
        await stream.aclose()

    asyncio.run(cancel())
    assert closed == [True]


@pytest.mark.parametrize(
    "stream_options", [None, {"include_usage": False}, {"include_usage": True}]
)
def test_http_stream_preserves_request_identity_and_usage_contract(stream_options):
    class StreamClient:
        async def completion_stream(self, request, *, request_id, audio_format):
            assert request_id == "usage-http"
            yield CompletionStreamChunk(
                request_id=request_id,
                text="hello",
                usage=UsageInfo(
                    prompt_tokens=3,
                    completion_tokens=1,
                    total_tokens=4,
                    prompt_tokens_details={"cached_tokens": 0},
                ),
            )
            yield CompletionStreamChunk(request_id=request_id, finish_reason="stop")

    payload = {
        "messages": [{"role": "user", "content": "hello"}],
        "stream": True,
        "request_id": "usage-http",
    }
    if stream_options is not None:
        payload["stream_options"] = stream_options
    with TestClient(create_app(StreamClient(), model_name="mock")) as client:
        response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = [
        line[6:] for line in response.text.splitlines() if line.startswith("data: ")
    ]
    assert events[-1] == "[DONE]"
    chunks = [json.loads(event) for event in events[:-1]]
    assert all(chunk["id"] == "chatcmpl-usage-http" for chunk in chunks)
    if stream_options is None:
        assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
        assert chunks[-1]["usage"]["total_tokens"] == 4
    elif stream_options["include_usage"]:
        assert chunks[-1]["choices"] == []
        assert chunks[-1]["usage"]["prompt_tokens_details"] == {"cached_tokens": 0}
    else:
        assert all("usage" not in chunk for chunk in chunks)


def test_generate_http_forwards_cache_and_keeps_metadata():
    class ResultClient:
        async def completion(self, request, *, request_id, audio_format):
            return CompletionResult(
                request_id=request_id,
                text="world",
                weight_version="v1",
                usage=UsageInfo(
                    prompt_tokens=12,
                    completion_tokens=2,
                    total_tokens=14,
                    prompt_tokens_details={"cached_tokens": 7},
                ),
            )

    with TestClient(create_app(ResultClient(), model_name="mock")) as client:
        response = client.post(
            "/generate",
            json={
                "prompt": "hello",
                "return_logprob": False,
                "metadata": {"group_id": 1},
            },
        )
    assert response.status_code == 200
    meta = response.json()["meta_info"]
    assert meta["cached_tokens"] == 7
    assert meta["prompt_tokens"] == 12
    assert meta["completion_tokens"] == 2
    assert meta["weight_version"] == "v1"
    assert meta["request_metadata"] == {"group_id": 1}
