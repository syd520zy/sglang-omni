# Metrics

Install the optional dependency with `pip install -e '.[metrics]'` and pass
`--enable-metrics` (or `--enable_metrics`) to `sgl-omni serve`. Metrics are
disabled by default. Scrape `GET /metrics` on the service port. The endpoint
has no API-key requirement; restrict access through deployment networking.

## SGLang-compatible core metrics

| Metric | Type | Unit | Labels |
| --- | --- | --- | --- |
| `sglang:e2e_request_latency_seconds` | Histogram | seconds | model_name, is_streaming |
| `sglang:time_to_first_token_seconds` | Histogram | seconds | model_name, is_streaming |
| `sglang:inter_token_latency_seconds` | Histogram | seconds | model_name |
| `sglang:prompt_tokens_total` | Counter | tokens | model_name, is_streaming |
| `sglang:generation_tokens_total` | Counter | tokens | model_name, is_streaming |
| `sglang:num_running_reqs` | Gauge | requests | model_name, stage |
| `sglang:num_queue_reqs` | Gauge | requests | model_name, stage |

Latency buckets follow the reference SGLang checkout (commit
`e310eaeaf766c207533b920f459d3d57263c5d12`). Names and types are compatible;
the multi-stage measurement boundary is necessarily Omni-specific.

- E2E starts at Client generation entry, before request construction, and ends
  when the final pipeline output reaches the Client. It includes preprocessing
  and downstream stages, but not HTTP admission before Client entry or final
  HTTP response encoding/transmission. It is not a single-stage engine timer.
- TTFT and ITL use dedicated host-side AR token events after SGLang updates
  `output_ids`. The Coordinator observes them for streaming and non-streaming
  requests alike; SSE framing and empty terminal chunks are not observations.
  IPC delivery is included, as in a tokenizer-side observation. No device
  synchronization or per-token device timestamp is added.
- The first AR stage to report tokens owns the request's token observations.
  Later stages are not added to it. These are raw backend token counts, not
  estimates derived from text length, image count or diffusion steps.
- For an interval producing N new tokens, ITL records N samples of interval/N.
  Duplicate cumulative counts do not add samples; a decrease resets the
  baseline without recording a negative increment.
- Token counters are recorded once when generation completes. If AR events
  exist, use their final count; otherwise use available final text usage.
  Missing usage produces no token sample. Pipeline exceptions and cancelled
  streams do not publish successful final usage. Explicit backend aborts are
  counted as completed/aborted requests with the available partial token count.
- Models without AR token telemetry, including image diffusion, have E2E
  observations but no fabricated TTFT/ITL observations.
- Each supported stage reports its actual scheduler state through the existing
  admin control plane. SimpleScheduler includes serial, batched and concurrent
  work; OmniScheduler includes active prefill/decode requests and its waiting
  queue. Preprocessing/admission backlog outside that queue is not folded into
  the SGLang queue gauge. Unsupported schedulers produce no synthetic zero.
- TP follower counts are not summed. Stage replicas have distinct stage labels.
  A request may exist in multiple stages: summing stage gauges is stage work,
  not a deduplicated service request count. Select the model execution stage
  when reusing a dashboard designed for a single SGLang engine.

## Export lifecycle and additional HTTP metrics

One HTTP process owns one app-local registry; generation workers report over
the control plane. Repeated app creation does not register global collectors.
There is no multiprocess file directory to clean. `PROMETHEUS_MULTIPROC_DIR`
is rejected before model loading because it changes Prometheus client storage;
unset it before launch. Multiple HTTP workers sharing one pipeline are not
supported by this exporter. Do not merge independently exported SRT metrics
into this registry: that would duplicate names/counts.

Scrapes refresh scheduler gauges. A timeout or failed stage returns HTTP 503,
not a successful response carrying stale values. Removed stages disappear on
the next successful scrape. Token observation state is removed on success,
failure and cancellation; late events are ignored.

Additional counters are `sglang:num_requests_total`,
`sglang:num_aborted_requests_total` and `sglang_omni:failed_requests_total`.
HTTP transport metrics are separate: `sglang_omni:http_requests_total`,
`sglang_omni:http_request_duration_seconds`, and
`sglang_omni:http_requests_inflight`. They cover validation errors and streams
through the final response body/disconnect. An HTTP 200 SSE may still contain
a model error; HTTP status is not model success. Routes use templates (or
`unmatched`), never request IDs, prompts or arbitrary paths. Health, favicon,
metrics scrapes and WebSockets are excluded.

## Aggregation

AVG/P95/P99 are queries over Histogram samples, not separate exporter gauges.
For example, a model's ITL average over five minutes is:

```promql
sum by (model_name) (rate(sglang:inter_token_latency_seconds_sum[5m]))
/
sum by (model_name) (rate(sglang:inter_token_latency_seconds_count[5m]))
```

P95 is `histogram_quantile(0.95, sum by (le, model_name)
(rate(sglang:inter_token_latency_seconds_bucket[5m])))`; use 0.99 for P99.
Token counters are lifetime totals; use `increase(...[5m])` for a period count.
Histogram percentiles are bucket-based estimates. No observations means no
valid latency estimate, not zero latency.

GPU/NPU latency overhead, actual TP/replica deployment and monitoring-platform
label compatibility require hardware/deployment acceptance. SenseNova Images
acceptance covers E2E and scheduler gauges, not text TTFT/ITL correctness.
Additional SGLang metrics (KV cache, cache hits, speculative decoding, PD,
queue-time histograms, TPOT, modality throughput) remain a later extension.
