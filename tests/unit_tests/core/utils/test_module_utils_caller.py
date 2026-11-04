###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""The rank-filtered logging helpers of primus.core.utils.module_utils name the code that called them
(module, function, line), which the logger prints as the record's origin."""

import sys

from primus.core.utils import logger, module_utils


def test_helpers_report_their_caller(monkeypatch):
    seen = []
    for name in ("info_with_caller", "debug_with_caller", "warning_with_caller", "error_with_caller", "log_kv_with_caller"):
        monkeypatch.setattr(logger, name, lambda *args, _n=name: seen.append((_n, args[-3:])), raising=False)
    monkeypatch.setattr(module_utils, "_rank", 0)
    monkeypatch.setattr(module_utils, "_world_size", 1)
    here = __name__.split(".")[-1]

    def emitter():
        lines = []
        module_utils.log_rank_0("a"); lines.append(sys._getframe().f_lineno)  # noqa: E702
        module_utils.log_rank_last("b"); lines.append(sys._getframe().f_lineno)  # noqa: E702
        module_utils.log_rank_all("c"); lines.append(sys._getframe().f_lineno)  # noqa: E702
        module_utils.log_kv_rank_0("k", "v"); lines.append(sys._getframe().f_lineno)  # noqa: E702
        module_utils.debug_rank_0("d"); lines.append(sys._getframe().f_lineno)  # noqa: E702
        module_utils.debug_rank_all("e"); lines.append(sys._getframe().f_lineno)  # noqa: E702
        module_utils.warning_rank_0("f"); lines.append(sys._getframe().f_lineno)  # noqa: E702
        module_utils.error_rank_0("g"); lines.append(sys._getframe().f_lineno)  # noqa: E702
        return lines

    lines = emitter()
    assert len(seen) == 8
    assert [s[1] for s in seen] == [(here, "emitter", line) for line in lines]
    assert [s[0] for s in seen] == [
        "info_with_caller",
        "info_with_caller",
        "info_with_caller",
        "log_kv_with_caller",
        "debug_with_caller",
        "debug_with_caller",
        "warning_with_caller",
        "error_with_caller",
    ]
