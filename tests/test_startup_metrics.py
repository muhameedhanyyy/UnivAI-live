import pytest

from telemetry.startup_metrics import summarize, summarize_audible


def test_percentiles_require_real_sample_count_and_preserve_failures():
    with pytest.raises(ValueError, match="30"):
        summarize([{"mode": "cold", "outcome": "ready", "total_ms": 1}])
    traces = []
    for mode, base in (("cold", 1000), ("warm", 500)):
        traces.extend({"mode": mode, "outcome": "ready", "total_ms": base + n} for n in range(29))
        traces.append({"mode": mode, "outcome": "failed", "total_ms": 8000})
    result = summarize(traces)
    assert result["cold"]["samples"] == 30 and result["cold"]["failures"] == 1
    assert result["warm"]["p95_ms"] == 527


def test_audible_summary_uses_browser_playback_events():
    events = []
    for mode, base in (("cold", 3000), ("warm", 1000)):
        events.extend({
            "schema": "univai.live.client-first-audio",
            "mode": mode,
            "client_elapsed_ms": base + n,
        } for n in range(30))
    result = summarize_audible(events)
    assert result["cold"]["p95_ms"] == 3028
    assert result["warm"]["p95_ms"] == 1028
