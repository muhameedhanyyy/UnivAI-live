# Startup performance evidence

The worker emits `univai.live.startup-trace` v1 with dispatch, room-connected, metadata-valid, artifact-loaded, track-published, ready-acknowledged and first-frame monotonic offsets. Summaries refuse to calculate until both cold and warm groups contain at least 30 raw traces.

Target-hardware evidence has not yet been recorded in this repository, so the warm-ready p95 ≤2 s and cold-ready p95 ≤5 s gates remain **NOT RUN**. Capture 30 cold and 30 warm configured sessions, retain the raw JSONL plus machine/network/model configuration, and summarize them with `telemetry.startup_metrics.summarize`. A missing artifact or startup timeout must become a learner-visible failure within 8 seconds.
