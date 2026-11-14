###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""The embedding-wgrad-into-main_grad patch (``embedding_wgrad_fusion_patches``): when it applies (CPU), and on GPU
the fused backward against the unfused one (dense ``embedding_dense_backward`` gradient + Megatron's
``main_grad.add_``): bitwise equal (bf16 and fp32 main_grad, mock-like and heavily repeated ids, small tables with
column tails), forward bitwise, deterministic, the weight's gradient the pooled dummy."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

import primus.backends.megatron.patches.embedding_wgrad_fusion_patches as patch

# ------------------------------------------------------------------------------------------------- selection (CPU)


def _ctx(**args):
    return SimpleNamespace(extra={"module_config": SimpleNamespace(params=SimpleNamespace(**args))})


@pytest.mark.parametrize(
    "args, want",
    [({}, False), (dict(embedding_wgrad_fusion=False), False), (dict(embedding_wgrad_fusion=True), True)],
)
def test_yaml_flag(args, want):
    assert patch._patch_condition(_ctx(**args)) is want


def _emb(device="cpu", main_grad_dtype=torch.float32, **kw):
    w = torch.nn.Parameter(torch.randn(64, 16, device=device).bfloat16())
    w.main_grad = torch.zeros(64, 16, device=device, dtype=main_grad_dtype)
    w.grad_added_to_main_grad = False
    for k in [k for k in kw if k.startswith("w_")]:
        setattr(w, k[2:], kw.pop(k))
    base = dict(
        weight=w,
        training=True,
        deterministic_mode=False,
        tp_group=SimpleNamespace(size=lambda: 1),
        gtp_remat_size=1,
        reduce_scatter_embeddings=False,
        config=SimpleNamespace(gradient_accumulation_fusion=True),
    )
    return SimpleNamespace(**{**base, **kw})


IDS = torch.tensor([[1, 5, 5, 63], [0, 5, 7, 1]])


def test_fusable_in_the_supported_case():
    assert patch._fusable(_emb(), IDS)


@pytest.mark.parametrize(
    "kw",
    [
        dict(training=False),
        dict(deterministic_mode=True),  # Megatron embeds with weight[ids] there
        dict(tp_group=SimpleNamespace(size=lambda: 2)),
        dict(gtp_remat_size=2),
        dict(reduce_scatter_embeddings=True),
        dict(config=SimpleNamespace(gradient_accumulation_fusion=False)),
        dict(w_zero_out_wgrad=True),  # tied output layer / MTP on this stage
    ],
)
def test_not_fusable(kw):
    assert not patch._fusable(_emb(**kw), IDS)


def test_not_fusable_without_main_grad_long_ids_or_deterministic_mode_attribute():
    emb = _emb()
    del emb.weight.main_grad
    assert not patch._fusable(emb, IDS)
    assert not patch._fusable(_emb(), IDS.int())
    with torch.no_grad():
        assert not patch._fusable(_emb(), IDS)
    emb = _emb()
    del emb.deterministic_mode  # fails closed
    assert not patch._fusable(emb, IDS)


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


V, H, B, S = 129408, 7168, 2, 4096  # a large vocabulary and hidden size, T = 8192 tokens


def _ids(kind, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    if kind == "random":  # mock data: few repeats
        return torch.randint(0, V, (B, S), device="cuda", generator=g)
    hot = torch.randint(0, V, (97,), device="cuda", generator=g)  # many repeats per row
    return hot[torch.randint(0, 97, (B, S), device="cuda", generator=g)]


def _setup(kind, main_grad_dtype, seed=0):
    """(weight, main_grad before, ids, grad_out) of one micro-batch."""
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


def _unfused(weight, main0, ids, grad):
    """(main_grad, forward output): F.embedding's dense gradient, then Megatron's grad hook main_grad.add_(grad)."""
    w = torch.nn.Parameter(weight.clone())
    out = F.embedding(ids, w)
    out.backward(grad)
    return main0.clone().add_(w.grad.data), out.detach()


@gpu
@pytest.mark.parametrize("main_grad_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("kind", ["random", "hot"])
def test_fused_is_bitwise_the_unfused_path(kind, main_grad_dtype):
    weight, main0, ids, grad = _setup(kind, main_grad_dtype)
    fused, out_f, w = _fused(weight, main0, ids, grad)
    unfused, out_u = _unfused(weight, main0, ids, grad)
    assert torch.equal(out_f, out_u), "the forward must be F.embedding's"
    assert w.grad_added_to_main_grad and w.grad.shape == w.shape
    untouched = torch.ones(V, dtype=torch.bool, device="cuda")
    untouched[torch.unique(ids)] = False
    assert torch.equal(fused[untouched], main0[untouched])
    assert torch.equal(fused, unfused), "the fused backward must be bitwise the unfused main_grad"


@gpu
def test_deterministic():
    """Repeated runs are bitwise identical."""
    case = _setup("hot", torch.bfloat16, seed=5)
    ref = _fused(*case)[0]
    assert torch.equal(_fused(*case)[0], ref)


@gpu
def test_weight_grad_is_the_pooled_dummy():
    """The weight's .grad is the dummy the dummy-wgrad helper hands out for its shape (not a copy), whatever the
    installed pool looks like."""
    w = _fused(*_setup("random", torch.bfloat16))[-1]
    assert w.grad.data_ptr() == patch._dummy_wgrad(w.shape, w.dtype).data_ptr()


@gpu
def test_flat_dummy_pool_does_not_grow():
    """With a Primus-Turbo whose dummy-wgrad pool is one flat buffer per (dtype, device), the embedding's dummy is
    a view of it: the buffer neither grows nor moves."""
    turbo_utils = pytest.importorskip("primus_turbo.pytorch.ops.utils")
    turbo_utils._get_dummy_wgrad([V + 1024, H], torch.bfloat16)  # a dummy larger than the table
    key = (torch.bfloat16, torch.cuda.current_device())
    pool = getattr(turbo_utils, "_dummy_wgrads", {}).get(key)
    if pool is None:
        pytest.skip("this Primus-Turbo keeps one dummy per shape")
    numel, ptr = pool.numel(), pool.data_ptr()
    w = _fused(*_setup("random", torch.bfloat16))[-1]
    pool = turbo_utils._dummy_wgrads[key]
    assert pool.numel() == numel and pool.data_ptr() == ptr, "the pool grew"
    assert w.grad.data_ptr() == ptr
