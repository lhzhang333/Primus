###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Which optional kernels the Primus-Turbo in use has, asked without knowing its file layout.

A Primus-Turbo with ``primus_turbo.capabilities`` answers itself. An older one is probed the way
Primus did before that module existed: two gfx1250 package directories and one function of the
gfx1250 grouped GEMM module. Nothing here imports torch, so the launch scripts can ask from the shell:

    python3 -m primus.core.utils.turbo_capability mega_moe.bf16     # prints True or False
    python3 -m primus.core.utils.turbo_capability --where           # the primus_turbo package directory
"""

import argparse
import importlib
import importlib.util
import os
from functools import lru_cache

# What a Primus-Turbo without primus_turbo.capabilities is probed for. It has these on gfx1250 only.
_LEGACY_ARCH = "gfx1250"
_LEGACY_DIRS = {
    "attention.flydsl.mha": ("flydsl", "attention", "gfx1250"),
    "mega_moe.bf16": ("flydsl", "mega", "gfx1250"),
}
_LEGACY_ATTRS = {
    "grouped_gemm.flydsl.bf16.wgrad_accum": (
        "primus_turbo.flydsl.grouped_gemm.grouped_gemm_bf16_kernel_gfx1250",
        "grouped_gemm_bf16_variable_k_accum_supported",
    ),
}


def turbo_package_dir() -> str:
    """The directory of the ``primus_turbo`` package Python would import, ``""`` without one.
    Does not import it."""
    try:
        spec = importlib.util.find_spec("primus_turbo")
    except (ImportError, ValueError):
        return ""
    if spec is None:
        return ""
    if spec.submodule_search_locations:
        return list(spec.submodule_search_locations)[0]
    return os.path.dirname(spec.origin) if spec.origin else ""


def _capabilities():
    """``primus_turbo.capabilities``, or None for a Primus-Turbo that predates it (or none at all)."""
    if not turbo_package_dir():
        return None
    try:
        if importlib.util.find_spec("primus_turbo.capabilities") is None:
            return None
    except (ImportError, ValueError):
        return None
    return importlib.import_module("primus_turbo.capabilities")


def _legacy_has(feature: str, arch: str) -> bool:
    if arch != _LEGACY_ARCH:
        return False
    if feature in _LEGACY_DIRS:
        package_dir = turbo_package_dir()
        return bool(package_dir) and os.path.isdir(os.path.join(package_dir, *_LEGACY_DIRS[feature]))
    if feature in _LEGACY_ATTRS:
        module, attr = _LEGACY_ATTRS[feature]
        try:
            return hasattr(importlib.import_module(module), attr)
        except ImportError:
            return False
    return False


@lru_cache(maxsize=None)
def turbo_has(feature: str, arch: str = "gfx1250") -> bool:
    """Whether the Primus-Turbo in use has ``feature`` for ``arch``. Cached: the package does not
    change under a running process (``turbo_has.cache_clear()`` forgets, for tests)."""
    capabilities = _capabilities()
    if capabilities is not None:
        return bool(capabilities.has(feature, arch))
    return _legacy_has(feature, arch)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("feature", nargs="?", help="e.g. mega_moe.bf16, attention.flydsl.mha")
    parser.add_argument("--arch", default="gfx1250")
    parser.add_argument("--where", action="store_true", help="print the primus_turbo package directory")
    args = parser.parse_args(argv)
    if args.where:
        print(turbo_package_dir())
    if args.feature:
        print(turbo_has(args.feature, args.arch))
    elif not args.where:
        parser.error("a feature name or --where is required")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
