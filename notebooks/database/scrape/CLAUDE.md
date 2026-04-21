# database/scrape/ — Integration Test Coverage

Scripts in this folder test that model-fit results can be scraped from disk into an
SQLite database via `Aggregator.from_database` + `agg.add_directory`, and that the
aggregator can query, reconstruct, and visualise those results.

Every script follows the same structure:
1. Load dataset + mask.
2. Build model + run `search.fit(...)`.
3. Scrape output directory into a fresh `.sqlite` file.
4. Test `agg.values("samples")`, `agg.model.*` queries, `agg.values("samples_summary")`.
5. Test the `TracerAgg`, `ImagingAgg`, and `FitImagingAgg` aggregator modules.

---

## general.py

**Model**: `Isothermal + ExternalShear` lens, `lp_linear.Sersic` source, one extra
galaxy with `lp_linear.Sersic` light (no mass scaling relation).

**Queries tested**:
- `unique_tag`, `search.name`
- `lens.mass == al.mp.Isothermal`
- `mass.einstein_radius > 1.0`
- `extra_galaxy.bulge == al.lp_linear.Sersic`

**Aggregator modules**: `TracerAgg`, `ImagingAgg`, `FitImagingAgg`

**Known issues**: None.

---

## Codex / sandboxed runs

When running Python from Codex or any restricted environment, set writable cache directories so `numba` and `matplotlib` do not fail on unwritable home or source-tree paths:

```bash
NUMBA_CACHE_DIR=/tmp/numba_cache MPLCONFIGDIR=/tmp/matplotlib python scripts/database/scrape/general.py
```

This workspace is often imported from `/mnt/c/...` and Codex may not be able to write to module `__pycache__` directories or `/home/jammy/.cache`, which can cause import-time `numba` caching failures without this override.

---

## multi_analysis.py

**Model**: Multi-stage fitting (two searches) over the same dataset.

Tests that the aggregator handles multi-analysis result directories correctly:
multiple `Search` results in one `add_directory` call, and `agg.query` filtering
by search name picks out the correct stage.

---

## slam_general.py

**Model**: Full four-stage SLaM pipeline (Source LP → Light LP → Mass Total) with
lens light (`DevVaucouleurs + Exponential` bulge/disk) and `PowerLaw` total mass.

Tests that the aggregator handles a chained multi-search output directory.
Queries use `agg.query(agg.search.name == "mass_total[1]")` to select the final
stage before running the aggregator modules.

---

## slam_pix.py

**Model**: SLaM pipeline with pixelized source reconstruction (Delaunay mesh).

Tests that `FitImagingAgg` correctly reconstructs pixelized fits from the database,
including `adapt_images` from the preceding adapt search.

---

## slam_multi_one_by_one.py

**Model**: SLaM pipeline run one-by-one over multiple datasets.

Tests that the aggregator can collate results from multiple independent model-fits
stored in separate output subdirectories.

---

## scaling_relation.py

**Model**: `Isothermal + ExternalShear` lens, `lp_linear.Sersic` source, one extra
galaxy whose Einstein radius is set via a luminosity scaling relation:

    einstein_radius = scaling_factor * luminosity ** scaling_exponent

`scaling_factor` and `scaling_exponent` are the two free parameters of the scaling
relation.  The luminosity (0.9) is a pre-measured fixed float.

This test deliberately exercises the serialise/deserialise round-trip for
`af.Model` objects that contain relational expressions — a known potential failure
point in the database layer.

**Queries tested**:
- Standard queries from `general.py`
- `extra_galaxy.mass == al.mp.Isothermal` on the scaling-relation extra galaxy

**Aggregator modules**: `TracerAgg`, `ImagingAgg`, `FitImagingAgg`

**Expected failure point**: The aggregator reconstruction step
(`samples_summary.max_log_likelihood()`, `TracerAgg`, `FitImagingAgg`) may fail
if the `scaling_factor * luminosity ** scaling_exponent` expression is not
correctly serialised into the database.  The test is structured to reach that point
and surface the exact error.
