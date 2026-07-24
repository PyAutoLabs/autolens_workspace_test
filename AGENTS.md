# PyAutoLens Workspace Test — Agent Instructions

This is the integration-test suite for **PyAutoLens**, run on the build server to verify the core
library works end-to-end. It is **not** a user-facing workspace — see `../autolens_workspace` for
examples and tutorials. These are the canonical, agent-agnostic instructions for this repo.

Dependencies: `autolens`, `autogalaxy`, `autofit`, `autoarray`, `numba`.

## Repository Structure

`scripts/` mirrors the `autolens_workspace` dataset taxonomy — dataset-typed
folders, plus `misc/` for everything dataset-agnostic. Within each dataset
folder, related tests are grouped into task subfolders — `jax_likelihood/`
(batched `_vmap` likelihood tests), `jax_grad/` (finite-difference gradient
tests), `visualization/`, `simulator/`, `substructure/` (imaging), `datacube/`
(interferometer) — with the dataset root holding its modeling singletons. The
former top-level `jax_likelihood_functions/`, `jax_grad/`, `jax_substructure/`,
`potential_correction/`, `model_composition/` and `light_multipole/` trees were
dissolved into these per-dataset subfolders.

```
scripts/                     Integration-test scripts run on the build server
  imaging/ interferometer/   CCD imaging / interferometer model-fit tests, with
                             jax_likelihood/ jax_grad/ visualization/ simulator/
                             (imaging: substructure/; interferometer: datacube/) subfolders
  point_source/ cluster/     Point-source and cluster model-fit tests
  multi/                     Multi-wavelength (FactorGraph) tests (jax_likelihood/ visualization/)
  misc/                      Dataset-agnostic tests:
    aggregator/ database/      Results database + aggregator tests
    jax_assertions/            JAX assertion tests
    mass/ mass_via_integral/   Mass-profile tests
    latent/ weak/ interop/     Latent-variable / weak-lensing / COOLEST-interop tests
    util.py                    Shared jax-gradient finite-difference helper
    hessian_jax.py profiles_jit.py tracer_jax.py tracer_multiplane.py ...  loose JAX/tracer tests
  gallery/ profiling/        Outside the taxonomy (external couplings — untouched)
failed/                      One .txt log per failing script (written by run_all_scripts.sh)
dataset/ config/ output/     Input data, YAML config, runtime fit results
```

Per-area detail lives in nested guides: `scripts/CLAUDE.md` and
`scripts/misc/database/scrape/CLAUDE.md` (sub-area references, not top-level docs).

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
`config/build/profile_smoke.yaml`. That file sets **fast-mode defaults for every entry** —
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

1. **JAX likelihood functions** (highest) — batched log-likelihood gradients via
   `fitness._vmap(parameters)` for the full `AnalysisImaging` + `Tracer` pipeline; one script per
   model type, distributed across `imaging/`, `interferometer/`, `point_source/` and `multi/`.
2. **`misc/hessian_jax.py`** (mid) — `LensCalc` hessian-derived quantities under the guard pattern; the
   **reference** for JAX testing style.
3. **`misc/tracer_jax.py`** (mid) — `Tracer` ray-tracing methods under `jax.jit` for two- and three-plane
   systems.
4. **`misc/profiles_jit.py`** (lowest) — individual light/mass profile methods under `jax.jit`.

Library unit tests stay NumPy-only; this repo is where the `xp=jnp` path is exercised. See the
PyAutoArray deep dive `../PyAutoArray/docs/agents/jax_and_decorators.md` for the boundary patterns.

## Bulk-edit safety

When editing the same region across many scripts in one pass, only rewrite the targeted region.
**Never produce a whole-file write unless you have read the entire current file** — a whole-file
write from a header skim silently deletes every section below the header.

## Related Repos

- Source libs: `../PyAutoLens`, `../PyAutoGalaxy`, `../PyAutoArray`, `../PyAutoFit`, `../PyAutoNerves`.
- `../autolens_workspace` — the user-facing workspace; `../HowToLens` — the tutorial series.
- `../PyAutoHands` — CI / build tooling.
- `../autolens_assistant` — science-assistant workspace (literature wiki).

## Task Workflows

When a library change lands, run the smoke suite, read any `[FAIL]` entries, and update the affected
test scripts to the new API (preserving intent). **Never edit a script to mask a real regression** —
if a library bug surfaces, flag it for the source repo rather than papering over it. Note in your PR
any change that affects sibling repos (`autolens_workspace`, the source libraries).

<!-- repos_sync:history:begin -->
## Never rewrite history

Never rewrite pushed history on any repo with a remote — no `git init` over a
tracked repo, no force-push to `main`, no fresh-start "Initial commit", no
`filter-repo` / `filter-branch` / `rebase -i` on pushed branches. To get a
clean tree: `git fetch origin && git reset --hard origin/main && git clean -fd`.
<!-- repos_sync:history:end -->
