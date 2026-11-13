# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""The embedding-wgrad-into-main_grad patch (``embedding_wgrad_fusion_patches``): when it applies (CPU), and on GPU
the fused backward against today's path (dense ``embedding_dense_backward`` gradient + Megatron's
``main_grad.add_``): bitwise equal (bf16 and fp32 main_grad, mock-like and heavily repeated ids, small tables with
tails), forward bitwise, deterministic, the dummy taken from the shared pool without growing it. Also prints how
far today's (and so the fused) result is from one rounding of the exact sum."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

import primus.backends.megatron.patches.embedding_wgrad_fusion_patches as patch

# ------------------------------------------------------------------------------------------------- selection (CPU)


@pytest.mark.parametrize(
    "env, arg, want", [(None, False, False), (None, True, True), ("1", False, True), ("0", True, False)]
)
def test_switch(monkeypatch, env, arg, want):
    if env is None:
        monkeypatch.delenv("PRIMUS_EMBEDDING_WGRAD_FUSION", raising=False)
    else:
        monkeypatch.setenv("PRIMUS_EMBEDDING_WGRAD_FUSION", env)
    assert patch._requested(SimpleNamespace(embedding_wgrad_fusion=arg)) is want


def _emb(device="cpu", main_grad_dtype=torch.float32, **kw):
    w = torch.nn.Parameter(torch.randn(64, 16, device=device).bfloat16())
    w.main_grad = torch.zeros(64, 16, device=device, dtype=main_grad_dtype)
    w.grad_added_to_main_grad = False
    for k in [k for k in kw if k.startswith("w_")]:
        setattr(w, k[2:], kw.pop(k))
    base = dict(weight=w, training=True, tp_group=SimpleNamespace(size=lambda: 1), gtp_remat_size=1,
                reduce_scatter_embeddings=False, config=SimpleNamespace(gradient_accumulation_fusion=True))  # fmt: skip
    return SimpleNamespace(**{**base, **kw})


IDS = torch.tensor([[1, 5, 5, 63], [0, 5, 7, 1]])


def test_fusable_in_the_supported_case():
    assert patch._fusable(_emb(), IDS)


@pytest.mark.parametrize(
    "kw",
    [
        dict(training=False),
        dict(tp_group=SimpleNamespace(size=lambda: 2)),
        dict(gtp_remat_size=2),
        dict(reduce_scatter_embeddings=True),
        dict(config=SimpleNamespace(gradient_accumulation_fusion=False)),
        dict(w_zero_out_wgrad=True),  # tied output layer / MTP on this stage
    ],
)
def test_not_fusable(kw):
    assert not patch._fusable(_emb(**kw), IDS)


def test_not_fusable_without_main_grad_or_long_ids():
    emb = _emb()
    del emb.weight.main_grad
    assert not patch._fusable(emb, IDS)
    assert not patch._fusable(_emb(), IDS.int())
    with torch.no_grad():
        assert not patch._fusable(_emb(), IDS)


def test_install_routes_and_falls_back(monkeypatch):
    from megatron.core.tensor_parallel import layers

    cls = layers.VocabParallelEmbedding
    monkeypatch.setattr(cls, "forward", lambda self, x: "orig")
    monkeypatch.setattr(patch, "is_patched", lambda *a: False)
    monkeypatch.setattr(patch, "mark_patched", lambda *a: None)
    patch._install_embedding_wgrad_fusion_patch()
    monkeypatch.setattr(patch, "_dummy_wgrad", lambda shape, dtype: torch.zeros(tuple(shape), dtype=dtype))
    emb = _emb()
    assert cls.forward(emb, IDS).shape == (2, 4, 16)
    assert cls.forward(_emb(w_zero_out_wgrad=True), IDS) == "orig"


# --------------------------------------------------------------------------------------------------- GPU numerics

gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU numerics")


@gpu
@pytest.mark.parametrize("main_grad_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("V_, H_, T_", [(64, 16, 9), (1000, 1100, 513), (50, 2048, 4096)])
def test_accumulate_rows_is_dense_backward_plus_add(V_, H_, T_, main_grad_dtype):
    """``accumulate_rows_`` = ``main.add_(embedding_dense_backward(rows, ids))``, bitwise (column tails, many repeats)."""
    g = torch.Generator(device="cuda").manual_seed(V_ + H_ + T_)
    ids = torch.randint(0, V_, (T_,), device="cuda", generator=g)
    rows = torch.randn(T_, H_, device="cuda", generator=g).bfloat16()
    main = (torch.randn(V_, H_, device="cuda", generator=g) * 1e-2).to(main_grad_dtype)
    want = main.clone().add_(torch.ops.aten.embedding_dense_backward(rows, ids, V_, -1, False))
    assert torch.equal(patch.accumulate_rows_(main.clone(), ids, rows), want)


V, H, B, S = 129408, 7168, 2, 4096  # DeepSeek-V3 (padded vocabulary), T = 8192 tokens


def _ids(kind, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    if kind == "random":  # mock data: few repeats
        return torch.randint(0, V, (B, S), device="cuda", generator=g)
    hot = torch.randint(0, V, (97,), device="cuda", generator=g)  # many repeats per row
    return hot[torch.randint(0, 97, (B, S), device="cuda", generator=g)]


def _ordered(t):
    """bf16 / fp32 bits as integers ordered like the values (for ulp distances across the sign)."""
    i = t.view(torch.int16 if t.dtype == torch.bfloat16 else torch.int32).long()
    sign = torch.iinfo(torch.int16 if t.dtype == torch.bfloat16 else torch.int32).min
    return torch.where(i < 0, sign - i, i)


def _ulp(a, b):
    return (_ordered(a) - _ordered(b)).abs()


def _setup(kind, main_grad_dtype, seed=0):
    """(weight, main_grad before, ids, grad_out) of a micro-batch at the DeepSeek-V3 shape."""
    g = torch.Generator(device="cuda").manual_seed(seed + 1)
    weight = (torch.randn(V, H, device="cuda", generator=g) * 0.02).bfloat16()
    main0 = (torch.randn(V, H, device="cuda", generator=g) * 1e-3).to(main_grad_dtype)
    grad = torch.randn(B, S, H, device="cuda", generator=g).bfloat16()
    return weight, main0, _ids(kind, seed + 2), grad


def _fused(weight, main0, ids, grad):
    """(main_grad, forward output, the weight) through the patch's Function."""
    w = torch.nn.Parameter(weight.clone())
    w.main_grad, w.grad_added_to_main_grad = main0.clone(), False
    out = patch._EmbeddingWgradIntoMainGrad.apply(ids, w)
    out.backward(grad)
    return w.main_grad, out.detach(), w


def _today(weight, main0, ids, grad):
    """(main_grad, forward output): F.embedding's dense gradient, then Megatron's grad hook main_grad.add_(grad)."""
    w = torch.nn.Parameter(weight.clone())
    out = F.embedding(ids, w)
    out.backward(grad)
    return main0.clone().add_(w.grad.data), out.detach()


@gpu
@pytest.mark.parametrize("main_grad_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("kind", ["random", "hot"])
def test_against_exact_sum_and_todays_path(kind, main_grad_dtype):
    weight, main0, ids, grad = _setup(kind, main_grad_dtype)
    fused, out_f, w = _fused(weight, main0, ids, grad)
    today, out_t = _today(weight, main0, ids, grad)
    assert torch.equal(out_f, out_t), "the forward must be F.embedding's"
    assert w.grad_added_to_main_grad and w.grad.shape == w.shape
    uniq, inv = torch.unique(ids, return_inverse=True)
    rows = torch.zeros(uniq.numel(), H, device="cuda", dtype=torch.float64)
    rows.index_add_(0, inv.flatten(), grad.reshape(-1, H).double())
    exact = (main0[uniq].double() + rows).to(main_grad_dtype)  # one rounding of the exact sum
    untouched = torch.ones(V, dtype=torch.bool, device="cuda")
    untouched[uniq] = False
    assert torch.equal(fused[untouched], main0[untouched]) and torch.equal(today[untouched], main0[untouched])
    uf, ut, ftd = _ulp(fused[uniq], exact), _ulp(today[uniq], exact), _ulp(fused[uniq], today[uniq])
    print(f"\nEMB {kind:6} {str(main_grad_dtype):14}: fused vs one-rounded exact: max {uf.max().item()} ulp, "
          f"{(uf != 0).float().mean().item():.2e} differ; today vs exact: max {ut.max().item()} ulp, "
          f"{(ut != 0).float().mean().item():.2e} differ; fused vs today: max {ftd.max().item()} ulp, "
          f"{(ftd != 0).float().mean().item():.2e} differ")  # fmt: skip
    assert torch.equal(fused, today), "the fused backward must be bitwise today's main_grad"


@gpu
def test_deterministic():
    """Repeated runs are bitwise identical (cross-process: scratch w_emb.py); under torch's deterministic-algorithms
    mode the patch stands down (Megatron's deterministic_mode embeds with weight[ids])."""
    case = _setup("hot", torch.bfloat16, seed=5)
    ref = _fused(*case)[0]
    assert torch.equal(_fused(*case)[0], ref)
    prev = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        assert not patch._fusable(_emb(device="cuda"), IDS.cuda())
    finally:
        torch.use_deterministic_algorithms(prev)


@gpu
def test_dummy_from_the_shared_pool_without_growing_it():
    pytest.importorskip("primus_turbo")
    from primus_turbo.pytorch.ops import utils as turbo_utils

    turbo_utils._get_dummy_wgrad(
        [V + 1024, H], torch.bfloat16
    )  # an expert stack's dummy, larger than the table
    pool = turbo_utils._dummy_wgrads[(torch.bfloat16, torch.cuda.current_device())]
    numel, ptr = pool.numel(), pool.data_ptr()
    w = _fused(*_setup("random", torch.bfloat16))[-1]
    pool = turbo_utils._dummy_wgrads[(torch.bfloat16, torch.cuda.current_device())]
    assert pool.numel() == numel and pool.data_ptr() == ptr, "the pool grew"
    assert w.grad.data_ptr() == ptr, "the weight's .grad is not the pooled dummy (copied?)"
