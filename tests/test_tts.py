import tts


def test_piper_defaults_to_cpu_without_probing_cuda(monkeypatch):
    monkeypatch.delenv("PIPER_DEVICE", raising=False)
    monkeypatch.setenv("DEVICE", "auto")
    monkeypatch.setattr(
        tts,
        "_onnx_cuda_provider_loadable",
        lambda: (_ for _ in ()).throw(AssertionError("CUDA probed")),
    )
    assert tts._use_piper_cuda() is False


def test_piper_uses_cuda_only_when_explicit_and_available(monkeypatch):
    monkeypatch.setenv("PIPER_DEVICE", "cuda")
    monkeypatch.setattr(tts, "cuda_available", lambda: True)
    monkeypatch.setattr(tts, "_onnx_cuda_provider_loadable", lambda: True)
    assert tts._use_piper_cuda() is True


def test_piper_falls_back_when_explicit_cuda_provider_is_broken(monkeypatch):
    monkeypatch.setenv("PIPER_DEVICE", "cuda")
    monkeypatch.setattr(tts, "cuda_available", lambda: True)
    monkeypatch.setattr(tts, "_onnx_cuda_provider_loadable", lambda: False)
    assert tts._use_piper_cuda() is False
