# Response usage

The client carries cumulative usage snapshots supplied by the output producer.
It does not sum snapshots across chunks or stages: a repeated prompt or a
cumulative completion count must not be charged twice. Producers remain
responsible for identifying the logical request's token counts. Cross-stage
accounting is not inferred by this adapter.

In addition to prompt, completion and total tokens, usage can carry optional
`prompt_tokens_details` (cached, image, audio and video token counts) and
`reasoning_tokens`. Missing details are omitted; an explicitly reported zero is
preserved. These fields do not enable reasoning or cache reporting in a model
that does not supply those statistics.

Nested `usage` fields take precedence over top-level fields. A reported
`cached_tokens`, either inside `usage` or at the top level, is mapped to
`prompt_tokens_details.cached_tokens` only when the nested details are absent.
The nested cache count takes precedence over a top-level cache count.
Existing total-token fallback behavior is
retained. Completion and speech results retain the last reported usage snapshot
when a subsequent terminal chunk contains no usage.

For recognized multi-terminal decode + audio results, the existing policy is
retained: use decode usage if available, otherwise use audio usage. Do not add
both token counts: audio tokens and text tokens may represent different work
and repeated input. A later usage snapshot replaces the preceding snapshot;
partial snapshots from unrelated stages are not merged field by field.

## Field ownership audit

| Surface | Source and policy | Boundary |
| --- | --- | --- |
| Qwen3-Omni text | Detokenizer uses prompt count and thinker output IDs | New detail fields are not measured automatically |
| Ming-Omni text | `build_text_usage` uses input IDs and thinker output IDs | No generic aggregation across stages |
| Multi-terminal text + audio | Decode usage, with audio usage as fallback | Never sum text and audio token counts |
| Completion / speech | Last reported cumulative usage snapshot | Empty terminal chunks do not erase usage |
| Chat HTTP | Basic counts and optional backend-provided details | Legacy basic count defaults remain unchanged |
| Native generate | Existing basic counts plus reported cached count | Missing cache count still uses legacy zero |
| Transcription | Existing interface-specific audio duration usage | Not converted to text token usage |
| Control plane | Existing message serialization carries nested usage | Serialization tested; real multiprocess execution still needs validation |

Request IDs, finish reasons, output logprobs, weight versions and rollout data
retain their existing interfaces. This change does not add new timing sources
or make unimplemented model metadata appear available.

The client exposes output token logprobs for native rollout generation. Chat
logprob objects also require token text and position-level information, which
the generic completion result does not currently provide; they are not
fabricated from the rollout carrier. Likewise, no new reasoning-content or
tool-call stream carrier is introduced by adding usage details.

## Chat streaming

With `stream_options: {"include_usage": true}`, the server emits the finish
chunk, a separate usage chunk with `choices: []`, then `[DONE]`. If the producer
does not report usage, the usage chunk contains `usage: null` rather than
invented token counts. With explicit `include_usage: false`, usage is omitted
from the stream.

For backward compatibility, omitting `stream_options` retains the existing
behavior of attaching available usage to the finish chunk. This differs from
the opt-in default used by some OpenAI-compatible servers.

## Native generate and timing

`/generate` forwards a reported cached-token count into `meta_info.cached_tokens`.
The existing default of zero is retained when no cache count is supplied;
consumers must not interpret that default as evidence of measured cache misses.
Changing this legacy default requires a separate compatibility decision.

`engine_time_s` remains a client-level statistic. It is not automatically
exposed as Chat usage or relabeled as end-to-end latency. No new request timers,
Prometheus collectors or metrics endpoint are introduced by this adaptation.

## Validation boundary

Focused tests cover usage round trips, zero versus missing details, nested
cache counts, cumulative snapshots, recognized multi-terminal precedence,
speech terminal chunks, control-plane serialization, Chat JSON/SSE, cancellation
and errors, request identity, and native generation metadata.

On Windows, native serving imports reach SGLang's Unix-only `resource` import.
CPU mock HTTP tests replace platform discovery only; this is not validation of
Linux deployment, GPU/NPU execution, model-owned statistics or real
multi-process workers. Run the focused tests in the normal Linux deployment
environment before claiming those paths are validated.
