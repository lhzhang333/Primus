###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""``te_dgrad_transposed_weight``: TE's NN dgrad GEMMs run as TN on a transposed weight.

Verifies:
  1. Only dgrad calls (layout NN, grad) on a plain contiguous 2-D BF16 / FP16 CUDA weight with nothing
     else attached go TN, with the weight transposed; every other call passes through unchanged.
  2. ``transpose_2d`` is ``x.t().contiguous()`` bit for bit.
  3. Through TE's ``general_gemm``, the TN dgrad matches the NN one.
"""

import pytest
import torch

pytest.importorskip("transformer_engine")

import primus.backends.megatron.patches.te_patches.dgrad_transposed_weight_patches as patch_mod

needs_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def _recorder(calls):
    def original(A, B, *args, **kwargs):
        calls.append((A, B, args, dict(kwargs)))
        return "out"

    return original


@needs_gpu
def test_routing():
    calls = []
    gemm = patch_mod._make_general_gemm(_recorder(calls), lambda a: a.t().contiguous())
    w = torch.randn(48, 32, device="cuda", dtype=torch.bfloat16)
    dy = torch.randn(16, 48, device="cuda", dtype=torch.bfloat16)
    assert gemm(w, dy, layout="NN", grad=True, out_dtype=torch.bfloat16) == "out"
    A, B, args, kw = calls[-1]
    assert kw["layout"] == "TN" and torch.equal(A, w.t()) and B is dy
    # pass-through: forward, wgrad, no grad flag, extra positional args, accumulate, quantized, fp32,
    # non-contiguous, a bias
    for a, kwargs in (
        (w, dict(layout="TN")),
        (w, dict(layout="NT", grad=True)),
        (w, dict(layout="NN", grad=False)),
        (w, dict(layout="NN", grad=True, accumulate=True)),
        (w, dict(layout="NN", grad=True, quantization_params=object())),
        (w.float(), dict(layout="NN", grad=True)),
        (w.t(), dict(layout="NN", grad=True)),
        (w, dict(layout="NN", grad=True, bias=torch.zeros(1, device="cuda"))),
    ):
        calls.clear()
        gemm(a, dy, **kwargs)
        assert calls[-1][0] is a and calls[-1][3] == kwargs
    calls.clear()
    gemm(w, dy, torch.bfloat16, layout="NN", grad=True)  # positional extras: TE's own call shape only
    assert calls[-1][0] is w and calls[-1][2] == (torch.bfloat16,)


@needs_gpu
@pytest.mark.parametrize("shape", [(28672, 4096), (4096, 14336), (6144, 4096), (4096, 4096), (3, 5), (0, 7)])
def test_transpose_2d(shape):
    from primus.backends.megatron.core.extensions._triton.transpose import transpose_2d

    x = torch.randn(*shape, device="cuda", dtype=torch.bfloat16)
    assert torch.equal(transpose_2d(x), x.t().contiguous())


@needs_gpu
@pytest.mark.parametrize("out_f,in_f", [(6144, 4096), (4096, 1408)])
def test_matches_nn_dgrad(out_f, in_f):
    from transformer_engine.pytorch.cpp_extensions import general_gemm

    from primus.backends.megatron.core.extensions._triton.transpose import transpose_2d

    torch.manual_seed(0)
    w = (torch.randn(out_f, in_f, device="cuda") * 0.02).to(torch.bfloat16)
    dy = torch.randn(512, out_f, device="cuda", dtype=torch.bfloat16)
    nn, *_ = general_gemm(w, dy, layout="NN", grad=True, out_dtype=torch.bfloat16)
    tn, *_ = patch_mod._make_general_gemm(general_gemm, transpose_2d)(
        w, dy, layout="NN", grad=True, out_dtype=torch.bfloat16
    )
    ref = dy.float() @ w.float()
    assert tn.shape == nn.shape == ref.shape
    torch.testing.assert_close(tn.float(), ref, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(tn, nn, rtol=2e-2, atol=2e-2)
