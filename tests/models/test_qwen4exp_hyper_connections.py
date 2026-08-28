"""qwen4exp hyper-connections against a literal transcription of the ggml graph.

The oracle below is written straight from llama.cpp ``src/models/qwen4exp.cpp``
(``build_hc_mix`` / ``build_hc_combine``), op for op in NumPy, deliberately without
looking at how the torch version factors things. Agreement between two independent
transcriptions of the same reference is the only check that means anything here: this
block wraps every sublayer, and getting it subtly wrong produces a model that runs at
full speed and emits fluent nonsense rather than one that crashes.

The specific things a shape check would not catch, each covered below:

* the RMS norm groups over one stream while its gamma spans all of them
* the ``1/hc`` scale lands before the SiLU, not after
* streams are collapsed by mean, not sum
* combine weights are ``2 * sigmoid(x / hc)``, centred on 1
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from freetoken.models.qwen4exp.hyper_connections import (
    grouped_rms_norm,
    hc_combine,
    hc_init,
    hc_mix,
)

HC = 4
D = 16
LOW_RANK = 6
T = 5
EPS = 1e-6


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _silu(x):
    return x * _sigmoid(x)


def ref_hc_mix(state, w_norm, w_down, w_up, w_inject):
    """Literal NumPy transcription of build_hc_mix.

    state    [T, hc, D]      (ggml: [n_embd, hc, nt])
    w_norm   [hc*D]
    w_down   [low_rank, hc*D]   (torch orientation; ggml stores [hc*D, low_rank])
    w_up     [hc*D, low_rank]
    w_inject [hc, hc*D]
    """
    hc_dim = HC * D
    # ggml_rms_norm reduces over ne[0] == n_embd, i.e. within one stream
    var = (state**2).mean(axis=-1, keepdims=True)
    xn = state / np.sqrt(var + EPS)
    xn = xn.reshape(T, hc_dim)
    xn = xn * w_norm                                   # ggml_mul with the [hc_dim] gamma

    lo = xn @ w_down.T                                 # build_lora_mm(w_down, xn)
    lo = _silu(lo * (1.0 / HC))                        # ggml_scale then ggml_silu
    gate = _sigmoid(lo @ w_up.T)                       # sigmoid(build_lora_mm(w_up, lo))

    gated = (xn * gate).reshape(T, HC, D)
    # collapse the streams by their mean: sum the hc views, then scale by 1/hc
    mixed = gated.sum(axis=1) * (1.0 / HC)

    inject = xn @ w_inject.T
    return mixed, inject


def ref_hc_combine(state, block_out, inject):
    """Literal NumPy transcription of build_hc_combine."""
    w = _sigmoid(inject * (1.0 / HC)) * 2.0            # centred on 1
    b = np.repeat(block_out[:, None, :], HC, axis=1)   # broadcast across streams
    return state + b * w[:, :, None]


@pytest.fixture
def weights():
    rng = np.random.default_rng(0)
    hc_dim = HC * D
    return {
        "state": rng.standard_normal((T, HC, D)).astype(np.float64),
        "w_norm": rng.uniform(0.5, 1.5, hc_dim).astype(np.float64),
        "w_down": rng.standard_normal((LOW_RANK, hc_dim)).astype(np.float64) * 0.1,
        "w_up": rng.standard_normal((hc_dim, LOW_RANK)).astype(np.float64) * 0.1,
        "w_inject": rng.standard_normal((HC, hc_dim)).astype(np.float64) * 0.1,
    }


def _torch(w, key):
    return torch.from_numpy(w[key]).to(torch.float64)


def test_mix_matches_the_ggml_reference(weights):
    want_mixed, want_inject = ref_hc_mix(
        weights["state"], weights["w_norm"], weights["w_down"],
        weights["w_up"], weights["w_inject"],
    )
    got_mixed, got_inject = hc_mix(
        _torch(weights, "state"), _torch(weights, "w_norm"), _torch(weights, "w_down"),
        _torch(weights, "w_up"), _torch(weights, "w_inject"), eps=EPS, hc=HC,
    )
    assert got_mixed.shape == (T, D)
    assert got_inject.shape == (T, HC)
    np.testing.assert_allclose(got_mixed.numpy(), want_mixed, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(got_inject.numpy(), want_inject, rtol=1e-10, atol=1e-10)


def test_combine_matches_the_ggml_reference(weights):
    rng = np.random.default_rng(1)
    block_out = rng.standard_normal((T, D))
    inject = rng.standard_normal((T, HC))
    want = ref_hc_combine(weights["state"], block_out, inject)
    got = hc_combine(
        _torch(weights, "state"),
        torch.from_numpy(block_out).to(torch.float64),
        torch.from_numpy(inject).to(torch.float64),
        hc=HC,
    )
    np.testing.assert_allclose(got.numpy(), want, rtol=1e-10, atol=1e-10)


def test_zero_injection_is_an_ordinary_residual_add(weights):
    """2*sigmoid centres the weights on 1, which is the reason for the factor of 2."""
    state = _torch(weights, "state")
    block_out = torch.zeros(T, D, dtype=torch.float64)
    block_out += 1.5
    got = hc_combine(state, block_out, torch.zeros(T, HC, dtype=torch.float64), hc=HC)
    np.testing.assert_allclose(got.numpy(), (state + 1.5).numpy(), rtol=1e-12, atol=1e-12)


def test_norm_groups_over_one_stream_not_the_flattened_state(weights):
    """A plain norm over hc*D would be a different function; this pins the grouping.

    With one stream scaled up and the others left alone, a grouped norm removes the scale
    within that stream only, so the other streams are untouched. A flattened norm would
    change every stream.
    """
    state = _torch(weights, "state").clone()
    ones = torch.ones(HC * D, dtype=torch.float64)
    base = grouped_rms_norm(state, ones, EPS, HC).reshape(T, HC, D)

    state[:, 0, :] *= 10.0
    scaled = grouped_rms_norm(state, ones, EPS, HC).reshape(T, HC, D)

    # stream 0 renormalises back to (almost) where it was; the rest do not move at all
    np.testing.assert_allclose(scaled[:, 1:, :].numpy(), base[:, 1:, :].numpy(),
                               rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(scaled[:, 0, :].numpy(), base[:, 0, :].numpy(),
                               rtol=1e-6, atol=1e-6)


def test_streams_are_averaged_not_summed(weights):
    """With identical streams the mix must land at the stream value, not hc times it."""
    one = _torch(weights, "state")[:, :1, :].expand(-1, HC, -1).contiguous()
    ones = torch.ones(HC * D, dtype=torch.float64)
    w_down = torch.zeros(LOW_RANK, HC * D, dtype=torch.float64)
    w_up = torch.zeros(HC * D, LOW_RANK, dtype=torch.float64)
    mixed, _ = hc_mix(one, ones, w_down, w_up, None, eps=EPS, hc=HC)
    # zero down/up => gate == sigmoid(0) == 0.5 everywhere, so mixed == 0.5 * normed stream
    normed = grouped_rms_norm(one, ones, EPS, HC).reshape(T, HC, D)
    np.testing.assert_allclose(mixed.numpy(), (0.5 * normed[:, 0, :]).numpy(),
                               rtol=1e-10, atol=1e-10)


def test_scale_before_silu_is_not_the_same_as_after():
    """Guards the 1/hc placement: SiLU is nonlinear, so the order is observable."""
    rng = np.random.default_rng(3)
    x = rng.standard_normal((T, LOW_RANK)) * 3.0
    before = _silu(x / HC)
    after = _silu(x) / HC
    assert not np.allclose(before, after, rtol=1e-3, atol=1e-3)


def test_init_replicates_the_embedding(weights):
    embed = torch.randn(T, D, dtype=torch.float64)
    state = hc_init(embed, HC)
    assert state.shape == (T, HC, D)
    for c in range(HC):
        np.testing.assert_array_equal(state[:, c, :].numpy(), embed.numpy())
