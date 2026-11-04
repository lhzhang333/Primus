###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""The output-layer backward on HipKittens TN kernels (``output_layer_gemm_hipkittens``) against
Megatron's ``linear_with_grad_accumulation_and_async_allreduce``, and the per-iteration weight
transposer it shares with the TE dgrad patch."""

from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def _hk():
    """((tn gemm, plan, supported), (nt gemm, plan, supported))"""
    try:
        from primus_turbo.hipkittens import gemm_gfx1250 as g
    except ImportError:
        pytest.skip("Primus-Turbo has no HipKittens gfx1250 GEMM")
    if not (hasattr(g, "hk_gemm_nt") and g.hk_gemm_available()):
        pytest.skip("HipKittens gfx1250 TN/NT GEMM unavailable here")
    return (g.hk_gemm_tn, g.hk_gemm_tn_plan, g.hk_gemm_tn_supported), (g.hk_gemm_nt, g.hk_gemm_nt_plan, g.hk_gemm_nt_supported)


def _megatron(input, weight):
    from megatron.core.tensor_parallel import layers

    return layers.linear_with_grad_accumulation_and_async_allreduce(
        input, weight, None, False, False, False, None, None, None, 1
    )


def _run(f, x, w, g):
    x = x.detach().clone().requires_grad_(True)
    w = torch.nn.Parameter(w.detach().clone())
    y = f(x, w)
    y.backward(g)
    return y.detach(), x.grad, w.grad


@pytest.mark.parametrize(
    "seq,hidden,vocab",
    [
        (4096, 1792, 8320),  # both backward GEMMs on HipKittens, a ragged N edge in the weight gradient
        (128, 512, 1000),  # HipKittens refuses both (too few tiles, K not a multiple of 128): matmul
    ],
)
def test_matches_megatron_bitwise_on_integer_data(seq, hidden, vocab):
    from primus.backends.megatron.core.extensions._triton.transpose import transpose_2d
    from primus.backends.megatron.patches.output_layer_hk_gemm_patches import _make_function

    tn, nt = _hk()
    transposes = []

    def transposed(w):
        transposes.append(tuple(w.shape))
        return transpose_2d(w.detach())

    fn = _make_function(tn, nt, transposed)
    gen = torch.Generator(device="cuda").manual_seed(0)

    def ints(*shape):
        return torch.randint(-2, 3, shape, device="cuda", generator=gen).to(torch.bfloat16)

    # small integers: every product and partial sum is exact in FP32, so any summation order agrees
    x, w, g = ints(seq, 2, hidden), ints(vocab, hidden), ints(seq, 2, vocab)
    ref = _run(_megatron, x, w, g)
    out = _run(fn.apply, x, w, g)
    for name, a, r in zip(("output", "grad_input", "grad_weight"), out, ref):
        assert a.shape == r.shape and a.dtype == r.dtype, name
        assert torch.equal(a, r), name
    on_hk = tn[1](hidden, 2 * seq, vocab) is not None
    assert transposes == ([(vocab, hidden)] if on_hk else [])


def test_random_data_close_to_megatron():
    from primus.backends.megatron.core.extensions._triton.transpose import transpose_2d
    from primus.backends.megatron.patches.output_layer_hk_gemm_patches import _make_function

    tn, nt = _hk()
    fn = _make_function(tn, nt, lambda w: transpose_2d(w.detach()))
    gen = torch.Generator(device="cuda").manual_seed(1)
    x = torch.randn(4096, 2, 1792, device="cuda", dtype=torch.bfloat16, generator=gen)
    w = torch.randn(8320, 1792, device="cuda", dtype=torch.bfloat16, generator=gen) * 0.02
    g = torch.randn(4096, 2, 8320, device="cuda", dtype=torch.bfloat16, generator=gen) * 1e-3
    for a, r in zip(_run(fn.apply, x, w, g), _run(_megatron, x, w, g)):
        err = (a.float() - r.float()).abs().max() / r.float().abs().max()
        assert err < 1e-2


def test_wrapper_keeps_megatron_for_other_cases(monkeypatch):
    _hk()
    from megatron.core.tensor_parallel import layers

    from primus.backends.megatron.patches import output_layer_hk_gemm_patches as mod

    calls = []

    def original(*args):
        calls.append(args)
        return "megatron"

    monkeypatch.setattr(layers, "linear_with_grad_accumulation_and_async_allreduce", original)
    mod.patch_output_layer_gemm_hipkittens(None)  # register_patch returns the function itself
    patched = layers.linear_with_grad_accumulation_and_async_allreduce
    assert patched is not original and patched._primus_hk_backward

    x = torch.zeros(4, 2, 8, device="cuda", dtype=torch.bfloat16)
    w = torch.nn.Parameter(torch.zeros(16, 8, device="cuda", dtype=torch.bfloat16))
    b = torch.zeros(16, device="cuda", dtype=torch.bfloat16)
    tp2 = SimpleNamespace(size=lambda: 2)
    for kwargs in (
        dict(bias=b),
        dict(gradient_accumulation_fusion=True),
        dict(allreduce_dgrad=True),
        dict(sequence_parallel=True),
        dict(grad_output_buffer=[]),
        dict(wgrad_deferral_limit=2),
        dict(tp_group=tp2),
        dict(gtp_remat_size=2),
    ):
        args = dict(
            input=x,
            weight=w,
            bias=None,
            gradient_accumulation_fusion=False,
            allreduce_dgrad=False,
            sequence_parallel=False,
            grad_output_buffer=None,
            wgrad_deferral_limit=None,
            tp_group=None,
            gtp_remat_size=1,
        )
        args.update(kwargs)
        assert patched(**args) == "megatron", kwargs
    assert len(calls) == 8
    # the plain case goes to the HipKittens-backward function
    y = patched(x, w, None, False, False, False, None, None, None, 1)
    assert isinstance(y, torch.Tensor) and y.shape == (4, 2, 16) and len(calls) == 8
    # applying the patch again keeps the one wrapper
    mod.patch_output_layer_gemm_hipkittens(None)
    assert layers.linear_with_grad_accumulation_and_async_allreduce is patched


def test_iteration_transposer_rewrites_in_place(monkeypatch):
    import megatron.training

    from primus.backends.megatron.core.extensions._triton.transpose import iteration_transposer

    args = SimpleNamespace(curr_iteration=3)
    monkeypatch.setattr(megatron.training, "get_args", lambda: args)
    transposed = iteration_transposer()
    w = torch.randn(300, 200, device="cuda", dtype=torch.bfloat16)
    t1 = transposed(w)
    assert torch.equal(t1, w.t())
    assert transposed(w) is t1  # same iteration: the same copy
    with torch.no_grad():
        w.mul_(2)  # the optimizer step
    assert torch.equal(transposed(w), t1) and not torch.equal(t1, w.t())  # still iteration 3
    args.curr_iteration = 4
    t2 = transposed(w)
    assert t2.data_ptr() == t1.data_ptr() and torch.equal(t2, w.t())  # rewritten in place
    other = torch.randn(64, 32, device="cuda", dtype=torch.bfloat16)
    assert torch.equal(transposed(other), other.t())
    args.curr_iteration = None
    assert torch.equal(transposed(w), w.t())  # outside training: a fresh copy
