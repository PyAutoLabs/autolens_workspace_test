#!/usr/bin/env bash
# Workspace-owned install epilogue for the reusable Smoke Tests workflow
# (PyAutoHeart/.github/workflows/smoke-tests.yml). Runs with cwd at the
# checkout root (the dependency chain is cloned beside `workspace/`) and
# receives PYTHON_VERSION. Everything that differs per workspace lives
# here; the ceremony lives in the reusable workflow.
set -e

pip install ./PyAutoConf ./PyAutoFit ./PyAutoArray ./PyAutoGalaxy ./PyAutoLens
pip install "jax<0.7" "jaxlib<0.7"
pip install "./PyAutoArray[optional]" "./PyAutoGalaxy[optional]" "./PyAutoLens[optional]"
pip install tensorflow-probability==0.25.0
# The [optional] re-resolution above can upgrade autonerves to the
# stale PyPI release (setuptools_scm reports the local copy as
# 1.0.dev0 from the shallow checkout). Pin the local source one
# last time so site-packages has skip_latents() and other recent
# autonerves APIs available at import time.
pip install --force-reinstall --no-deps ./PyAutoConf
