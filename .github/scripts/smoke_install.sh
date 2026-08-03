#!/usr/bin/env bash
# Workspace-owned install epilogue for the reusable Smoke Tests workflow
# (PyAutoHeart/.github/workflows/smoke-tests.yml). Runs with cwd at the
# checkout root (the dependency chain is cloned beside `workspace/`) and
# receives PYTHON_VERSION. Everything that differs per workspace lives
# here; the ceremony lives in the reusable workflow.
set -e

pip install ./PyAutoNerves ./PyAutoFit ./PyAutoArray ./PyAutoGalaxy ./PyAutoLens
pip install "jax<0.7" "jaxlib<0.7"
pip install "./PyAutoArray[optional]" "./PyAutoGalaxy[optional]" "./PyAutoLens[optional]"
# NOTE: do NOT `pip install tensorflow-probability==0.25.0` here. The stable
# release crashes at import under the resolved modern JAX
# (`jax.interpreters.xla.pytype_aval_mappings` was removed), which broke the
# JAX Matern-kernel (delaunay_mge) likelihood path. The working modified-Bessel
# dependency is `tfp-nightly`, pinned by `PyAutoArray[optional]` above.
# The [optional] re-resolution above can upgrade autonerves to the
# stale PyPI release (setuptools_scm reports the local copy as
# 1.0.dev0 from the shallow checkout). Pin the local source one
# last time so site-packages has skip_latents() and other recent
# autonerves APIs available at import time.
pip install --force-reinstall --no-deps ./PyAutoNerves

# --- THROWAWAY: provenance check for PyAutoNerves#146 ---------------------
# This branch exists only to prove the setup.py source-stamp fix end-to-end
# before the library PRs merge. It is closed, never merged.
#
# The right discriminator is the VERSION, not `__file__`: smoke installs
# non-editable, so every import resolves under site-packages either way. A
# family package reporting a date version means a PyPI wheel shadowed the
# local source build — green for the wrong reason.
python3 - <<'PY'
import importlib.metadata as md

EXPECTED = "9999.0.0.dev0"
bad = []
for pkg in ("autonerves", "autofit", "autoarray", "autogalaxy", "autolens"):
    try:
        version = md.version(pkg)
    except md.PackageNotFoundError:
        continue
    print(f"{pkg:12} {version}")
    if version != EXPECTED:
        bad.append(f"{pkg}=={version}")

if bad:
    raise SystemExit(
        "PROVENANCE FAIL — resolved from PyPI, not the source checkout: "
        + ", ".join(bad)
    )
print("provenance OK — every family package is the local source build")
PY
