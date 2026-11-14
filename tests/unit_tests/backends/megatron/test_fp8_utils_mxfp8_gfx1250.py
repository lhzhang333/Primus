###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""fp8_utils.get_fp8_context: on gfx1250 the MXFP8 recipe keeps TransformerEngine's autocast off (its MXFP8 recipe
does not run there), so only the Primus-Turbo modules quantize; other archs and recipes keep TE's autocast on. The
autocast itself is a stand-in, so any GPU works (fp8_utils imports TransformerEngine and Primus-Turbo)."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from tests.utils import skip_if_no_cuda

skip_if_no_cuda()

import primus.backends.megatron.core.fp8_utils as fp8_utils  # noqa: E402  isort:skip

pytestmark = pytest.mark.skipif(
    not (fp8_utils.HAVE_TE and fp8_utils.HAVE_TURBO), reason="needs TransformerEngine and Primus-Turbo"
)


@pytest.mark.parametrize(
    "recipe, gfx1250, te_enabled",
    [("mxfp8", True, False), ("mxfp8", False, True), ("tensorwise", True, True)],
)
def test_te_autocast_off_for_mxfp8_on_gfx1250(monkeypatch, recipe, gfx1250, te_enabled):
    from megatron.core.enums import Fp8Recipe

    import primus.backends.megatron.core.extensions.primus_turbo as ext
    import primus.core.utils.gpu_arch as gpu_arch

    seen = {}
    monkeypatch.setattr(
        ext, "primus_turbo_fp8_autocast", lambda **kwargs: seen.update(kwargs) or nullcontext()
    )
    monkeypatch.setattr(gpu_arch, "is_gfx1250", lambda: gfx1250)
    config = SimpleNamespace(
        fp8="e4m3",
        fp8_recipe=Fp8Recipe(recipe),
        fp8_param=False,
        fp8_wgrad=True,
        fp8_dot_product_attention=False,
        first_last_layers_bf16=False,
        num_layers=1,
        num_layers_at_start_in_bf16=0,
        num_layers_at_end_in_bf16=0,
        tp_only_amax_red=False,
    )
    fp8_utils.get_fp8_context(config)
    assert seen["enabled"] is te_enabled
    assert seen["enabled_turbo"] is True  # Primus-Turbo quantizes either way
