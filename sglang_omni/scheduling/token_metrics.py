# SPDX-License-Identifier: Apache-2.0
"""Build host-only token telemetry after SGLang updates request output IDs."""

from sglang_omni.scheduling.messages import OutgoingMessage


def token_metrics_message(req, *, is_entry_rank: bool, aborted: bool):
    payload = getattr(getattr(req, "omni_data", None), "stage_payload", None)
    if (
        not is_entry_rank
        or aborted
        or payload is None
        or not payload.request.metadata.get("omni_metrics_enabled")
        or not req.output_ids
    ):
        return None
    return OutgoingMessage(
        request_id=req.rid,
        type="metrics",
        data={
            "prompt_tokens": len(req.origin_input_ids),
            "completion_tokens": len(req.output_ids),
        },
    )
