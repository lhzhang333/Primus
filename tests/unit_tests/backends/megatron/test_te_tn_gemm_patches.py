###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""TE's dgrad / wgrad GEMMs through the TN-dgrad / HipKittens patch (``te_dgrad_tn``, ``te_gemm_hipkittens``,
``te_wgrad_hipkittens``) against TE's own ``general_gemm``.

Small-integer operands keep every FP32 partial sum exact, so any correct kernel gives TE's BF16 result:
the comparisons are bit for bit.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def _te_gemm():
    try:
        from transformer_engine.pytorch.cpp_extensions import general_gemm
    except ImportError:
        pytest.skip("Transformer Engine unavailable")
    return general_gemm


def _hk():
    try:
        from primus_turbo.hipkittens import gemm_gfx1250 as g
    except ImportError:
        pytest.skip("Primus-Turbo has no HipKittens gfx1250 GEMM")
    if not (hasattr(g, "hk_gemm_nt") and g.hk_gemm_available()):
        pytest.skip("HipKittens gfx1250 TN/NT GEMM unavailable here")
    return g


def _ints(*shape, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randint(-2, 3, shape, device="cuda", generator=g).to(torch.bfloat16)


def _make(original, dgrad_tn, hipkittens, wgrad_hk, transposes=None):
    from primus.backends.megatron.core.extensions._triton.transpose import transpose_2d
    from primus.backends.megatron.patches.te_patches.tn_gemm_patches import _make_general_gemm

    def transposed(w):
        if transposes is not None:
            transposes.append(tuple(w.shape))
        return transpose_2d(w.detach())

    return _make_general_gemm(original, dgrad_tn, hipkittens, transposed, wgrad_hk)


@pytest.mark.parametrize(
    "out_f,in_f,tokens,hipkittens",
    [
        (4096, 7168, 8192, True),  # HipKittens takes it (32 x 32 tiles of 224 x 256)
        (4096, 7168, 8192, False),  # hipBLASLt TN on the transposed weight
        (576, 7168, 8192, True),  # K = 576: HipKittens refuses; TE's NN call, no transpose
    ],
)
def test_dgrad_matches_te(out_f, in_f, tokens, hipkittens):
    original = _te_gemm()
    if hipkittens:
        _hk()
    transposes = []
    gemm = _make(original, True, hipkittens, False, transposes)
    w = _ints(out_f, in_f, seed=1)
    dy = _ints(tokens, out_f, seed=2)
    ref = original(w, dy, layout="NN", grad=True, out_dtype=torch.bfloat16)
    out = gemm(w, dy, layout="NN", grad=True, out_dtype=torch.bfloat16)
    assert len(out) == len(ref) and all(o is None for o in out[1:])
    assert torch.equal(out[0], ref[0])
    refused = hipkittens and out_f % 128 != 0
    assert transposes == ([] if refused else [(out_f, in_f)])


@pytest.mark.parametrize(
    "out_f,in_f,lead",
    [
        (7168, 4096, (8192,)),
        (16384 // 2, 7168, (8192,)),
        (1536, 7168, (8192,)),  # HipKittens refuses (too few tiles): TE's NT call
        (7168, 16384, (4096, 2)),  # x and dy with leading [seq, batch] dims, as TE Linear passes them
    ],
)
def test_wgrad_matches_te(out_f, in_f, lead):
    original = _te_gemm()
    _hk()
    gemm = _make(original, False, False, True)
    x = _ints(*lead, in_f, seed=3)
    dy = _ints(*lead, out_f, seed=4)
    ref = original(x, dy, layout="NT", grad=True, out_dtype=torch.bfloat16)
    out = gemm(x, dy, layout="NT", grad=True, out_dtype=torch.bfloat16)
    assert out[0].shape == (out_f, in_f) and all(o is None for o in out[1:])
    assert torch.equal(out[0], ref[0])


def test_other_calls_pass_through():
    calls = []

    def original(A, B, *args, **kwargs):
        calls.append(kwargs)
        return "te", None, None, None

    gemm = _make(original, True, False, False)
    w = _ints(256, 512, seed=5)
    dy = _ints(1024, 256, seed=6)
    for kwargs in (
        dict(layout="TN"),  # forward
        dict(layout="NN", grad=True, accumulate=True),
        dict(layout="NN", grad=True, bias=torch.zeros(512, device="cuda", dtype=torch.bfloat16)),
        dict(layout="NN", grad=True, out=torch.empty(1024, 512, device="cuda", dtype=torch.bfloat16)),
        dict(layout="NN", grad=True, out_dtype=torch.float32),
        dict(layout="NT", grad=True),  # wgrad without te_wgrad_hipkittens
    ):
        assert gemm(w, dy, **kwargs)[0] == "te"
    assert len(calls) == 6
    assert gemm(w.float(), dy.float(), layout="NN", grad=True)[0] == "te"  # not BF16


def test_iteration_transposer(monkeypatch):
    from types import SimpleNamespace

    import megatron.training

    from primus.backends.megatron.core.extensions._triton.transpose import iteration_transposer

    args = SimpleNamespace(curr_iteration=0)
    monkeypatch.setattr(megatron.training, "get_args", lambda: args)
    transposed = iteration_transposer()
    w = _ints(384, 640, seed=7)
    t0 = transposed(w)
    assert torch.equal(t0, w.t()) and transposed(w) is t0
    with torch.no_grad():
        w.add_(1)
    args.curr_iteration = 1
    t1 = transposed(w)
    assert t1.data_ptr() == t0.data_ptr() and torch.equal(t1, w.t())


def _main_grad(out_f, in_f, seed):
    """An integer BF16 [out_f, in_f] slice of a larger buffer, 128 bytes into it (a gradient buffer aligns
    parameters to 64 elements, not 256 bytes)."""
    buf = torch.zeros(out_f * in_f + 128, dtype=torch.bfloat16, device="cuda")
    view = buf[64 : 64 + out_f * in_f].view(out_f, in_f)
    view.copy_(_ints(out_f, in_f, seed=seed) * 16)
    return buf, view


@pytest.mark.parametrize(
    "out_f,in_f,lead",
    [
        (7168, 4096, (8192,)),
        (7168, 16384, (4096, 2)),
        (1536, 7168, (8192,)),  # HipKittens refuses (too few tiles): TE's accumulating NT call
    ],
)
def test_wgrad_into_main_grad_matches_te(out_f, in_f, lead):
    """Gradient-accumulation fusion: TE's wgrad GEMM adds into main_grad (out=main_grad, accumulate=True)."""
    original = _te_gemm()
    g = _hk()
    if not hasattr(g, "hk_gemm_nt_out_ok"):
        pytest.skip("this Primus-Turbo's NT kernel does not accumulate")
    gemm = _make(original, False, False, True)
    x = _ints(*lead, in_f, seed=7)
    dy = _ints(*lead, out_f, seed=8)
    _, ref = _main_grad(out_f, in_f, seed=9)
    buf, main_grad = _main_grad(out_f, in_f, seed=9)
    kwargs = dict(layout="NT", grad=True, out_dtype=torch.bfloat16, accumulate=True)
    original(x, dy, out=ref, **kwargs)
    out = gemm(x, dy, out=main_grad, **kwargs)
    assert out[0] is main_grad and all(o is None for o in out[1:])
    assert torch.equal(main_grad, ref)
    assert buf[:64].count_nonzero() == 0 and buf[64 + main_grad.numel() :].count_nonzero() == 0


def test_wgrad_into_main_grad_other_cases_pass_through():
    _hk()
    calls = []

    def original(A, B, *args, **kwargs):
        calls.append(kwargs)
        return "te", None, None, None

    gemm = _make(original, False, False, True)
    x = _ints(8192, 4096, seed=10)
    dy = _ints(8192, 7168, seed=11)
    fp32_grad = torch.zeros(7168, 4096, device="cuda", dtype=torch.float32)
    bf16_grad = torch.zeros(7168, 4096, device="cuda", dtype=torch.bfloat16)
    for kwargs in (
        dict(out=fp32_grad, out_dtype=torch.float32, accumulate=True),  # FP32 main_grad
        dict(out=bf16_grad, out_dtype=torch.bfloat16, accumulate=False),  # overwrite, not accumulate
        dict(out=bf16_grad, out_dtype=torch.bfloat16, accumulate=True, bias=torch.zeros(7168, device="cuda")),
    ):
        assert gemm(x, dy, layout="NT", grad=True, **kwargs)[0] == "te"
    assert len(calls) == 3
