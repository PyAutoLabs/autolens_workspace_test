# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Repo Is

**autolens_workspace_test** is the integration test suite for PyAutoLens. It contains Python scripts that are run on the build server to verify that the core PyAutoLens functionality works end-to-end. It is not a user-facing workspace — see `../autolens_workspace` for example scripts and tutorials.

Dependencies: `autolens`, `autogalaxy`, `autofit`, `numba`. Python version: 3.11.

## Workspace Structure

```
scripts/                     Integration test scripts run on the build server
  imaging/                   CCD imaging model-fit tests
  interferometer/            Interferometer model-fit tests
  point_source/              Point source model-fit tests
  jax_likelihood_functions/  JAX likelihood function tests (imaging, interferometer, point_source)
  hessian_jax.py             JAX JIT tests for LensCalc hessian-derived lensing quantities
  profiles_jit.py            JAX JIT tests for light and mass profile methods
  tracer_multiplane.py       Multi-plane ray-tracing logic correctness tests (NumPy only)
  tracer_jax.py              JAX JIT tests for Tracer ray-tracing methods
  database/scrape/           Database scrape tests (see scripts/database/scrape/CLAUDE.md)
failed/                      Failure logs written here when a script errors (one .txt per failure)
dataset/                     Input .fits files and example data
config/                      YAML configuration files
output/                      Model-fit results written here at runtime
```

For full coverage detail on each script see `scripts/CLAUDE.md`.
For database scrape test detail see `scripts/database/scrape/CLAUDE.md`.

## Running Tests

Scripts are run from the repository root **without** `PYAUTOFIT_TEST_MODE=1` — the non-linear searches run for real (using sampler limits like `n_like_max` to keep runtimes short):

```bash
python scripts/imaging/model_fit.py
```

**Codex / sandboxed runs**: when running from Codex or any restricted environment, set writable cache directories so `numba` and `matplotlib` do not fail on unwritable home or source-tree paths:

```bash
NUMBA_CACHE_DIR=/tmp/numba_cache MPLCONFIGDIR=/tmp/matplotlib python scripts/imaging/model_fit.py
```

This workspace is often imported from `/mnt/c/...` and Codex may not be able to write to module `__pycache__` directories or `/home/jammy/.cache`, which can cause import-time `numba` caching failures without this override.

To run all tests and log failures to `failed/`:

```bash
bash run_all_scripts.sh
```

Each failed script produces a `.txt` file in `failed/` named after the script path (with `/` replaced by `__`), containing the exit code and full output.

Unlike `../autolens_workspace`, there is no resume/skip logic — every run executes all scripts in `scripts/` from scratch.

## Integration Test Runner

`run_all_scripts.sh` at the repo root:
- Finds all `*.py` files under `scripts/` and runs them in order (no test mode flag)
- On failure: writes a log to `failed/<script_path_with_slashes_replaced>.txt`
- Does not skip previously-run scripts (stateless, always runs all)

## JAX Testing

There are four layers of JAX integration testing, each targeting a different level of the stack:

1. **`jax_likelihood_functions/`** — highest level.  Tests that JAX can compute
   batched log-likelihood gradients via `fitness._vmap(parameters)` for the full
   `AnalysisImaging` + `Tracer` pipeline.  One script per model type.

2. **`hessian_jax.py`** — mid level.  Tests `LensCalc` hessian-derived quantities
   (`convergence_2d_via_hessian_from`, `shear`, `magnification`, `jacobian`,
   eigenvalues) using the three-step JAX pattern (NumPy / JAX outer / JAX JIT).
   This is the **reference** for the JAX testing style.

3. **`tracer_jax.py`** — mid level.  Tests `Tracer.traced_grid_2d_list_from`,
   `image_2d_from`, `deflections_yx_2d_from`, `convergence_2d_from` under JAX JIT
   for two-plane and three-plane systems.

4. **`profiles_jit.py`** — lowest level.  Tests individual light profile
   `image_2d_from` and mass profile `deflections_yx_2d_from` / `convergence_2d_from`
   methods under JAX JIT.  Covers `lp.Sersic`, `lp.Exponential`, `lp.Gaussian`,
   `lp.DevVaucouleurs`, `mp.Isothermal`, `mp.PowerLaw`, `mp.NFW`, `mp.ExternalShear`.

## Line Endings — Always Unix (LF)

All files **must use Unix line endings (LF, `\n`)**. Never write `\r\n` line endings.
## Never rewrite history

NEVER perform these operations on any repo with a remote:

- `git init` in a directory already tracked by git
- `rm -rf .git && git init`
- Commit with subject "Initial commit", "Fresh start", "Start fresh", "Reset
  for AI workflow", or any equivalent message on a branch with a remote
- `git push --force` to `main` (or any branch tracked as `origin/HEAD`)
- `git filter-repo` / `git filter-branch` on shared branches
- `git rebase -i` rewriting commits already pushed to a shared branch

If the working tree needs a clean state, the **only** correct sequence is:

    git fetch origin
    git reset --hard origin/main
    git clean -fd

This applies equally to humans, local Claude Code, cloud Claude agents, Codex,
and any other agent. The "Initial commit — fresh start for AI workflow" pattern
that appeared independently on origin and local for three workspace repos is
exactly what this rule prevents — it costs ~40 commits of redundant local work
every time it happens.
