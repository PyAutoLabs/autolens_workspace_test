# PyAutoLens Workspace Test — Agent Instructions

This is the integration-test suite for **PyAutoLens**, run on the build server to verify the core
library works end-to-end. It is **not** a user-facing workspace — see `../autolens_workspace` for
examples and tutorials. These are the canonical, agent-agnostic instructions for this repo.

Dependencies: `autolens`, `autogalaxy`, `autofit`, `autoarray`, `numba`.

## Repository Structure

```
scripts/                     Integration-test scripts run on the build server
  imaging/ interferometer/   CCD imaging / interferometer model-fit tests
  point_source/ cluster/     Point-source and cluster model-fit tests
  aggregator/ database/      Results database + aggregator tests
  jax_likelihood_functions/  JAX batched-likelihood tests (imaging, interferometer, point_source)
  jax_grad/ jax_assertions/  JAX gradient + assertion tests
  mass/ multi/ latent/       Mass, multi-wavelength, latent-variable tests
failed/                      One .txt log per failing script (written by run_all_scripts.sh)
dataset/ config/ output/     Input data, YAML config, runtime fit results
```

Per-area detail lives in nested guides: `scripts/CLAUDE.md` and
`scripts/database/scrape/CLAUDE.md` (left in place — sub-area references, not top-level docs).

## Running Scripts

Run a single script directly from the repo root — with no env applied, the non-linear search runs
for real (sampler limits like `n_like_max` keep it short):

```bash
python scripts/imaging/model_fit.py
```

## Testing

On CI, `smoke_tests.yml` gates every PR on Python **3.12 and 3.13**. The gate runs the smoke runner
(the definition of green):

```bash
python .github/scripts/run_smoke.py
```

It executes the curated entries in `smoke_tests.txt`, applying per-entry environment from
`config/build/env_vars.yaml`. That file sets **fast-mode defaults for every entry** —
`PYAUTO_TEST_MODE=2` (skip the sampler, structural/end-to-end check only), `PYAUTO_SMALL_DATASETS=1`
(cap grids/masks), `PYAUTO_FAST_PLOTS=1` — with **per-script `unset`/override** blocks where a test
genuinely needs a real sampler run or full-resolution data. So CI is *not* "searches run for real"
by default; it is fast-mode with targeted exceptions. A script that fails under these flags is a
real problem (broken import, renamed API, etc.).

For a local **full sweep** of every script under `scripts/` (not just the smoke subset), use the
stateless runner, which logs each failure to `failed/<script_path>.txt`:

```bash
bash run_all_scripts.sh
```

## Sandboxed / restricted runs

If `numba` or `matplotlib` cannot write to the default cache locations, point them at writable dirs:

```bash
NUMBA_CACHE_DIR=/tmp/numba_cache MPLCONFIGDIR=/tmp/matplotlib python scripts/imaging/model_fit.py
```

## JAX Testing

Four layers of JAX integration testing, each targeting a different level of the stack:

1. **`jax_likelihood_functions/`** (highest) — batched log-likelihood gradients via
   `fitness._vmap(parameters)` for the full `AnalysisImaging` + `Tracer` pipeline; one script per
   model type.
2. **`hessian_jax.py`** (mid) — `LensCalc` hessian-derived quantities under the guard pattern; the
   **reference** for JAX testing style.
3. **`tracer_jax.py`** (mid) — `Tracer` ray-tracing methods under `jax.jit` for two- and three-plane
   systems.
4. **`profiles_jit.py`** (lowest) — individual light/mass profile methods under `jax.jit`.

Library unit tests stay NumPy-only; this repo is where the `xp=jnp` path is exercised. See the
PyAutoArray deep dive `../PyAutoArray/docs/agents/jax_and_decorators.md` for the boundary patterns.

## Bulk-edit safety

When editing the same region across many scripts in one pass, only rewrite the targeted region.
**Never produce a whole-file write unless you have read the entire current file** — a whole-file
write from a header skim silently deletes every section below the header.

## Related Repos

- Source libs: `../PyAutoLens`, `../PyAutoGalaxy`, `../PyAutoArray`, `../PyAutoFit`, `../PyAutoConf`.
- `../autolens_workspace` — the user-facing workspace; `../HowToLens` — the tutorial series.
- `../PyAutoBuild` — CI / build tooling.
- `../autolens_assistant` — science-assistant workspace (literature wiki).

## Task Workflows

When a library change lands, run the smoke suite, read any `[FAIL]` entries, and update the affected
test scripts to the new API (preserving intent). **Never edit a script to mask a real regression** —
if a library bug surfaces, flag it for the source repo rather than papering over it. Note in your PR
any change that affects sibling repos (`autolens_workspace`, the source libraries).

<!-- repos_sync:history:begin -->
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
<!-- repos_sync:history:end -->
