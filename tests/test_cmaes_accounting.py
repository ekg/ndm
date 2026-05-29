import pytest


pytest.importorskip("cma")
pytest.importorskip("tiktoken")


def test_cmaes_estimator_uses_tokenizer_vocab(monkeypatch):
    import scripts.cmaes_search_v2 as cmaes
    from scripts.calc_dim import calc_e88_params

    params = {
        "dim": 2048,
        "depth": 10,
        "n_heads": 348,
        "n_state": 32,
        "expansion": 1.0,
    }

    monkeypatch.setattr(cmaes, "PARAM_VOCAB_SIZE", 256)
    byte_count = cmaes.estimate_params_for_config(params, "e88")

    vocab_size = cmaes.resolve_vocab_size("p50k_base")
    monkeypatch.setattr(cmaes, "PARAM_VOCAB_SIZE", vocab_size)
    bpe_count = cmaes.estimate_params_for_config(params, "e88")

    assert vocab_size == 50281
    assert bpe_count - byte_count == (vocab_size - 256) * params["dim"]
    assert bpe_count == calc_e88_params(**params, vocab_size=vocab_size)


def test_gdn2_wrapper_accepts_n_heads_alias():
    pytest.importorskip("fla")

    from pathlib import Path

    if not Path("/home/erikg/GatedDeltaNet-2/lit_gpt/gdn2.py").exists():
        pytest.skip("external GDN-2 checkout is not available")

    from ndm.models.external_gdn2 import GDN2ExternalLayer

    layer = GDN2ExternalLayer(dim=128, expansion=1, head_dim=16, n_heads=3)

    assert layer.num_heads == 3
    assert layer.gdn2.num_heads == 3
