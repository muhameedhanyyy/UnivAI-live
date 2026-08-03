# Demo resilience runbook

Run `python demo/preflight.py` before the demonstration and the externally configured smoke Q&A separately. Exercise a normal raised-hand question, then make Agent and TTS unavailable and verify that the learner sees the fallback reason while the lecture resumes.

Latency targets are STT 8 s, retrieval/generation 15 s, TTS 5 s and total Q&A 30 s. These are budgets, not measured evidence. Capture representative traces on the target device; no target-device measurement is committed yet. A recorded artifact may be used only for an external outage, must be disclosed, and must be registered in `demo/fallback-manifest.json`.
