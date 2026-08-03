import pytest

from telemetry.startup_metrics import summarize


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
