# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest

from sglang_omni.scheduling.token_metrics import token_metrics_message


@pytest.mark.parametrize(
    "enabled,leader,aborted",
    [(False, True, False), (True, False, False), (True, True, True)],
)
def test_disabled_follower_and_abort_do_not_emit(enabled, leader, aborted):
    req = request(enabled)
    assert token_metrics_message(req, is_entry_rank=leader, aborted=aborted) is None


def request(enabled=True):
    return SimpleNamespace(
        rid="r",
        origin_input_ids=[1, 2, 3],
        output_ids=[4, 5],
        omni_data=SimpleNamespace(
            stage_payload=SimpleNamespace(
                request=SimpleNamespace(metadata={"omni_metrics_enabled": enabled}),
            )
        ),
    )


def test_telemetry_is_cumulative_and_does_not_mutate_output():
    req = request()
    message = token_metrics_message(req, is_entry_rank=True, aborted=False)
    assert message.type == "metrics"
    assert message.data == {"prompt_tokens": 3, "completion_tokens": 2}
    assert req.output_ids == [4, 5]
    req.output_ids.clear()
    assert token_metrics_message(req, is_entry_rank=True, aborted=False) is None
