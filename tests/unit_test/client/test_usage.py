# SPDX-License-Identifier: Apache-2.0
"""Usage carriers preserve backend statistics without manufacturing details."""

import asyncio

import pytest

from sglang_omni.client import Client
from sglang_omni.client.types import GenerateChunk, GenerateRequest, UsageInfo
from sglang_omni.pipeline.control_plane import deserialize_message, serialize_message
from sglang_omni.proto import StreamMessage


def test_usage_details_round_trip():
    data = {
        "prompt_tokens": 12,
        "completion_tokens": 3,
        "total_tokens": 15,
        "prompt_tokens_details": {"cached_tokens": 0, "image_tokens": 8},
        "reasoning_tokens": 0,
    }
    assert UsageInfo.from_dict(data).to_dict() == data


def test_missing_details_are_not_reported_as_zero():
    usage = UsageInfo(prompt_tokens=12).to_dict()
    assert "prompt_tokens_details" not in usage
    assert "reasoning_tokens" not in usage


def test_backend_cache_count_and_nested_precedence():
    usage = Client.build_usage_info({"cached_tokens": 5, "prompt_tokens": 12})
    assert usage.prompt_tokens_details == {"cached_tokens": 5}
    usage = Client.build_usage_info(
        {"cached_tokens": 5, "usage": {"prompt_tokens_details": {"cached_tokens": 0}}}
    )
    assert usage.prompt_tokens_details == {"cached_tokens": 0}


def test_completion_preserves_usage_before_empty_terminal_chunk():
    class ChunkClient(Client):
        async def generate(self, request, *, request_id):
            yield GenerateChunk(
                request_id=request_id,
                text="hello",
                usage=UsageInfo(prompt_tokens=12, completion_tokens=1, total_tokens=13),
            )
            yield GenerateChunk(request_id=request_id, finish_reason="stop")

    result = asyncio.run(
        ChunkClient(None).completion(
            GenerateRequest(prompt="hello"), request_id="usage"
        )
    )
    assert result.usage.total_tokens == 13
    assert result.text == "hello"


def test_nested_cache_count_is_preserved():
    usage = Client.build_usage_info(
        {"usage": {"prompt_tokens": 12, "cached_tokens": 5}}
    )
    assert usage.prompt_tokens_details == {"cached_tokens": 5}


def test_repeated_cumulative_usage_is_not_added():
    class ChunkClient(Client):
        async def generate(self, request, *, request_id):
            for count in (1, 1, 2):
                yield GenerateChunk(
                    request_id=request_id,
                    usage=UsageInfo(
                        prompt_tokens=12,
                        completion_tokens=count,
                        total_tokens=12 + count,
                    ),
                )
            yield GenerateChunk(request_id=request_id, finish_reason="stop")

    result = asyncio.run(
        ChunkClient(None).completion(
            GenerateRequest(prompt="hello"), request_id="usage"
        )
    )
    assert result.usage.total_tokens == 14


def test_multi_terminal_usage_prefers_decode_without_summing():
    chunk = Client.default_result_builder(
        "merged",
        {
            "decode": {
                "text": "hello",
                "usage": {
                    "prompt_tokens": 12,
                    "completion_tokens": 2,
                    "total_tokens": 14,
                },
            },
            "talker": {
                "usage": {
                    "prompt_tokens": 12,
                    "completion_tokens": 20,
                    "total_tokens": 32,
                }
            },
        },
    )
    assert chunk.usage.total_tokens == 14


def test_speech_preserves_usage_before_empty_terminal_chunk():
    class ChunkClient(Client):
        async def generate(self, request, *, request_id):
            yield GenerateChunk(
                request_id=request_id,
                audio_data=[0.0, 0.1],
                sample_rate=24000,
                usage=UsageInfo(prompt_tokens=12, completion_tokens=2, total_tokens=14),
            )
            yield GenerateChunk(request_id=request_id, finish_reason="stop")

    result = asyncio.run(
        ChunkClient(None).speech(
            GenerateRequest(prompt="hello"), request_id="speech", response_format="wav"
        )
    )
    assert result.usage.total_tokens == 14
    assert result.audio_bytes.startswith(b"RIFF")


def test_completion_does_not_return_partial_usage_on_failure():
    class ChunkClient(Client):
        async def generate(self, request, *, request_id):
            yield GenerateChunk(request_id=request_id, usage=UsageInfo(total_tokens=14))
            raise RuntimeError("backend failed")

    with pytest.raises(RuntimeError, match="backend failed"):
        asyncio.run(
            ChunkClient(None).completion(
                GenerateRequest(prompt="hello"), request_id="failed"
            )
        )


def test_usage_survives_control_plane_serialization():
    usage = UsageInfo(
        prompt_tokens=12,
        completion_tokens=2,
        total_tokens=14,
        prompt_tokens_details={"cached_tokens": 5, "audio_tokens": 8},
        reasoning_tokens=0,
    )
    message = StreamMessage(
        request_id="transport",
        from_stage="decode",
        chunk={"usage": usage.to_dict()},
        modality="text",
    )
    decoded = deserialize_message(serialize_message(message))
    chunk = Client.default_stream_builder("transport", decoded)
    assert chunk.usage.to_dict() == usage.to_dict()
