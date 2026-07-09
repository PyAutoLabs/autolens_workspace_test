"""
Visualization: Cluster Strong Lens
===================================

Integration test for the **cluster-scale aplt helpers** (`autolens/cluster/plot/`) against the
cluster simulator output. This script's previous life was the raw-matplotlib *prototype* those
helpers were promoted from (PyAutoLens#577); it now exercises the promoted library API end to end
and asserts each expected PNG is written to disk.

Dataset: ``autolens_workspace_test/dataset/cluster/test/`` (written by
``scripts/cluster/simulator.py``, fed by the CSV truth model from ``scripts/cluster/csv_api.py``).
If absent, both helper scripts are invoked at the start of this script and the dataset is
regenerated at the workspace_test resolution (500x500 @ 0.1"/px).

Structure
---------
1. Resolve the cluster dataset path inside this workspace_test repo. Auto-simulate if missing.
2. Load ``data.fits``, ``point_datasets.csv``, ``tracer.json`` and read the lens / halo centres
   from ``mass.csv``.
3. Run five visualization phases, each writing one PNG into a clean
   ``scripts/cluster/images/visualization/`` directory:

     - ``visualization_overlaid_positions.png`` — ``aplt.plot_positions_overlay``: full-field
       cluster image (percentile LogNorm + ``gnuplot2``), every source's positions in the Wong
       (2011) palette, BCG/halo markers, kpc scale bar.
     - ``visualization_image_zooms.png`` — ``aplt.plot_image_group_zooms``: one zoom panel per
       multiple image, rows grouped and colour-framed by source.
     - ``visualization_critical_curves.png`` — ``aplt.plot_critical_curves``: tangential + radial
       critical curves of **every source plane** (per-plane colours + redshift legend) over the
       cluster image. This supersedes the prototype's single ``plane_j=-1`` curve — at cluster
       scale the z=1 and z=2 source planes have visibly different critical curves.
     - ``visualization_caustics.png`` — ``aplt.plot_caustics``: the per-plane tangential + radial
       caustics in source-plane coordinates.
     - ``visualization_subplot_dataset.png`` — ``aplt.subplot_cluster_dataset``: positions overlay
       | per-plane critical curves mosaic.

4. Assert each PNG exists with non-zero size, and that at least one tangential critical curve is
   recovered per source plane for the 10^15.3 M_sun host.

Run from the ``autolens_workspace_test`` repo root with the standard cache overrides::

    NUMBA_CACHE_DIR=/tmp/numba_cache MPLCONFIGDIR=/tmp/matplotlib \\
        python scripts/cluster/visualization.py
"""

import shutil
import subprocess
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import autolens as al
import autolens.plot as aplt
from autogalaxy.operate.lens_calc import LensCalc


"""
__Paths__

The cluster test dataset lives in ``autolens_workspace_test/dataset/cluster/test/`` (written by
``scripts/cluster/simulator.py`` in this same workspace). ``parents[2]`` walks
``visualization.py → cluster/ → scripts/`` and ``parents[3]`` points at the workspace root.
"""
WORKSPACE_PATH = Path(__file__).resolve().parents[2]
DATASET_PATH = WORKSPACE_PATH / "dataset" / "cluster" / "test"
CSV_API_PATH = WORKSPACE_PATH / "scripts" / "cluster" / "csv_api.py"
SIMULATOR_PATH = WORKSPACE_PATH / "scripts" / "cluster" / "simulator.py"

PIXEL_SCALE = 0.1


"""
__Dataset Auto-Simulation__

If the dataset is absent (or missing ``data.fits``), run ``csv_api.py`` then ``simulator.py``. Both
scripts write paths *relative to their own CWD*, so we run them with ``cwd=WORKSPACE_PATH``.
"""
if not (DATASET_PATH / "data.fits").exists():
    print(
        f"Cluster test dataset missing at {DATASET_PATH} — running csv_api + simulator..."
    )
    subprocess.run([sys.executable, str(CSV_API_PATH)], cwd=WORKSPACE_PATH, check=True)
    subprocess.run(
        [sys.executable, str(SIMULATOR_PATH)], cwd=WORKSPACE_PATH, check=True
    )
    print("Cluster simulator complete.")


"""
__Load Dataset__

Four pieces are needed:

 - ``data.fits`` for the cluster background image.
 - ``point_datasets.csv`` — one ``PointDataset`` per source, carrying that source's image-plane
   positions and redshift.
 - ``tracer.json`` — the true ``Tracer``; gives the per-plane critical curves / caustics and the
   cosmology / lens redshift used by the kpc scale bar.
 - ``mass.csv`` — named-galaxy mass profiles; provides the lens / halo centres for the panel
   overlays (BCG, satellite, host-halo markers).
"""
data = al.Array2D.from_fits(
    file_path=DATASET_PATH / "data.fits", pixel_scales=PIXEL_SCALE
)
point_datasets = al.list_from_csv(file_path=DATASET_PATH / "point_datasets.csv")
tracer = al.from_json(file_path=DATASET_PATH / "tracer.json")

mass_table = al.galaxy_models_from_csv(DATASET_PATH / "mass.csv", family="mass")
_centres_by_galaxy = {row.galaxy: row.params["centre"] for row in mass_table.rows}
main_lens_centres = al.Grid2DIrregular(
    [
        _centres_by_galaxy[name]
        for name in ("lens_0", "lens_1")
        if name in _centres_by_galaxy
    ]
)
host_halo_centre = al.Grid2DIrregular([_centres_by_galaxy["host_halo"]])

positions_list = [dataset.positions for dataset in point_datasets]


"""
__Output Paths__

Mirrors the imaging visualization integration test: clean a per-script ``images/`` directory on
each run so assertions reflect this run only.
"""
image_path = Path("scripts") / "cluster" / "images" / "visualization"

if image_path.exists():
    shutil.rmtree(image_path)
image_path.mkdir(parents=True)


def assert_png(filename: str):
    path = image_path / filename
    assert path.exists(), f"{filename} missing"
    assert path.stat().st_size > 0, f"{filename} is empty"
    print(f"{filename} OK")


"""
__Plot 1 — Overlaid Positions (aplt.plot_positions_overlay)__

The default per-source ``subplot_point_dataset`` puts each source on its own axes, so multiple-image
groups belonging to different sources can never be compared spatially against the cluster image.
The promoted helper renders the cluster image once and overlays every source's positions in a
distinct palette colour, with the conventional markers and a kpc scale bar.
"""
print("Running plot 1 — overlaid positions (aplt.plot_positions_overlay)...")
_t0 = time.perf_counter()

aplt.plot_positions_overlay(
    positions_list,
    image=data,
    centres=main_lens_centres,
    halo_centres=host_halo_centre,
    redshift=float(tracer.galaxies[0].redshift),
    cosmology=tracer.cosmology,
    kpc_scale_bar=50.0,
    output_path=str(image_path),
    output_filename="visualization_overlaid_positions",
    output_format="png",
)

print(f"plot 1 complete in {time.perf_counter() - _t0:.2f}s")
assert_png("visualization_overlaid_positions.png")


"""
__Plot 2 — Per-Image Zoom Grid (aplt.plot_image_group_zooms)__

One zoom panel per observed multiple image, colour-framed by source. Full-field residual plots hide
a half-arcsecond position mismatch that a 6" zoom makes obvious.
"""
print("\nRunning plot 2 — image zoom grid (aplt.plot_image_group_zooms)...")
_t0 = time.perf_counter()

aplt.plot_image_group_zooms(
    positions_list,
    image=data,
    zoom_arcsec=8.0,
    output_path=str(image_path),
    output_filename="visualization_image_zooms",
    output_format="png",
)

print(f"plot 2 complete in {time.perf_counter() - _t0:.2f}s")
assert_png("visualization_image_zooms.png")


"""
__Plot 3 — Per-Plane Critical Curves (aplt.plot_critical_curves)__

The tangential critical curve at cluster scale is *not* one tidy ellipse — the massive host halo
produces a large outer curve while member galaxies produce small inner curves; and each source
plane has its *own* curve set (D_LS / D_S differs per plane). The promoted helper draws every
source plane's curves in that source's palette colour, labelled by redshift.

The 0.2"/px viz grid over the full 50" field resolves the host-halo-scale curves; see the helper
docstring for grid guidance at arcminute scale.
"""
print("\nRunning plot 3 — per-plane critical curves (aplt.plot_critical_curves)...")
_t0 = time.perf_counter()

viz_grid = al.Grid2D.uniform(shape_native=(250, 250), pixel_scales=0.2)

aplt.plot_critical_curves(
    tracer,
    grid=viz_grid,
    image=data,
    include_radial=True,
    output_path=str(image_path),
    output_filename="visualization_critical_curves",
    output_format="png",
)

print(f"plot 3 complete in {time.perf_counter() - _t0:.2f}s")
assert_png("visualization_critical_curves.png")


"""
__Plot 4 — Per-Plane Caustics (aplt.plot_caustics)__
"""
print("\nRunning plot 4 — per-plane caustics (aplt.plot_caustics)...")
_t0 = time.perf_counter()

aplt.plot_caustics(
    tracer,
    grid=viz_grid,
    output_path=str(image_path),
    output_filename="visualization_caustics",
    output_format="png",
)

print(f"plot 4 complete in {time.perf_counter() - _t0:.2f}s")
assert_png("visualization_caustics.png")


"""
__Plot 5 — Combined Mosaic (aplt.subplot_cluster_dataset)__
"""
print("\nRunning plot 5 — combined mosaic (aplt.subplot_cluster_dataset)...")
_t0 = time.perf_counter()

aplt.subplot_cluster_dataset(
    positions_list,
    image=data,
    tracer=tracer,
    grid=viz_grid,
    centres=main_lens_centres,
    halo_centres=host_halo_centre,
    output_path=str(image_path),
    output_filename="visualization_subplot_dataset",
    output_format="png",
)

print(f"plot 5 complete in {time.perf_counter() - _t0:.2f}s")
assert_png("visualization_subplot_dataset.png")


"""
__Physics Assertion — Per-Plane Curves Recovered__

At least one tangential critical curve must be recovered for **each** source plane of the
10^15.3 M_sun host — and the outer (higher-redshift) plane's curve must enclose a larger area than
the closer plane's, since it sees a larger D_LS / D_S.
"""
median_radii = []
for j in range(1, len(tracer.planes)):
    lens_calc = LensCalc.from_tracer(tracer, use_multi_plane=True, plane_j=j)
    curves = lens_calc.tangential_critical_curve_list_from(grid=viz_grid)
    assert len(curves) >= 1, (
        f"no tangential critical curves recovered for plane {j} "
        f"(z={float(tracer.planes[j].redshift):.2f})"
    )
    import numpy as np

    radii = [
        float(np.median(np.linalg.norm(np.asarray(curve.array), axis=1)))
        for curve in curves
    ]
    median_radii.append(max(radii))
    print(
        f"plane {j} (z={float(tracer.planes[j].redshift):.2f}): "
        f'{len(curves)} tangential curve(s), outermost median radius {max(radii):.2f}"'
    )

assert median_radii == sorted(median_radii), (
    "higher-redshift source planes must have larger tangential critical curves "
    f"(got median radii {median_radii})"
)

print("\nCluster visualization integration test complete — all assertions passed.")
