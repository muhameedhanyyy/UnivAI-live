# Startup performance evidence

The worker emits `univai.live.startup-trace` v1 with dispatch, room-connected,
metadata-valid, artifact-loaded, track-published, ready-acknowledged and server
first-frame monotonic offsets. The browser additionally emits
`univai.live.client-first-audio` when its audio element fires `playing`. Use that
browser event for the learner-audible SLO; server first-frame is diagnostic only.
Summaries require at least 30 raw samples in both cold and warm groups.

Target-hardware evidence has not yet been recorded, so the warm-audible p95 <=2 s
and cold-audible p95 <=5 s gates remain **NOT RUN**.

1. Set `STARTUP_TRACE_PATH=/var/log/univai/startup.jsonl` and `STARTUP_MODE=warm`.
2. Pre-render the target lecture with `python prerender_audio.py --student <id> --week <n>`, keep the worker and cache hot, and launch 30 fresh browser rooms.
3. Set `STARTUP_MODE=cold`, restart the worker with an empty disposable audio cache for every run, and launch 30 more rooms.
4. Retain the JSONL and machine/network/model configuration. Call `telemetry.startup_metrics.summarize_audible` for the 2 s/5 s gates and `summarize` for server-stage diagnosis.
5. Inject missing-artifact, unavailable-TTS, and unreachable-storage cases and assert the page shows its error or sound-blocked warning by 8 seconds.

The browser timer begins before the token request and ends at the `playing` event,
so it includes the learner-visible App, dispatch, network, and playback path.
