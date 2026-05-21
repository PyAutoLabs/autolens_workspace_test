"""
Visualization: Cluster Strong Lens
===================================

Integration test that exercises three cluster-scale visualization patterns against the cluster
simulator output and asserts each expected PNG is written to disk. The patterns are built with raw
``matplotlib.pyplot`` + ``al.LensCalc`` only — **no** ``Visuals2D``, ``Include2D``, or
``aplt.subplot_*`` is touched, by design. Library-side plotter changes are deliberately deferred to
a follow-up prompt informed by what this prototype surfaces.

Dataset: ``autolens_workspace_test/dataset/cluster/test/`` (written by
``scripts/cluster/simulator.py``, fed by the CSV truth model from ``scripts/cluster/csv_api.py``).
If absent, both helper scripts are invoked at the start of this script and the dataset is
regenerated at the workspace_test resolution (500x500 @ 0.1"/px).

Structure
---------
1. Resolve the cluster dataset path inside this workspace_test repo. Auto-simulate if missing.
2. Load ``data.fits``, ``point_datasets.csv``, ``tracer.json`` and read the lens / halo centres
   from ``mass.csv``.
3. Run three visualization phases, each writing one PNG into a clean
   ``scripts/cluster/images/visualization/`` directory:

     - ``visualization_overlaid_positions.png`` — full-field cluster image (LogNorm + ``gnuplot2``)
       with every source's positions overlaid in the Wong (2011) colour-blind palette, BCG/halo
       markers, a kpc scale bar, and a per-source legend.
     - ``visualization_per_source_grid.png`` — one zoomed panel per source, each cropped to that
       source's image-group bounding box plus a 30 %-of-span margin (5" floor).
     - ``visualization_critical_curves.png`` — full-field cluster image with the multi-plane
       (``plane_j=-1``) tangential critical curve overlaid in cyan at cluster-tuned line weight
       and opacity.

4. Assert each PNG exists with non-zero size.

Why no ``Visuals2D`` here
-------------------------
The current ``Visuals2D`` API cannot express per-source colouring (every position is rendered in
the same colour), per-image-group zooms, or scale bars. The galaxy-scale critical-curve defaults
also overwhelm a 100" field. Encoding cluster patterns with raw ``matplotlib`` here keeps the
research deliverable separate from any library-side change, so the follow-up prompt can decide
which patterns are worth promoting into ``aplt`` from a working reference.

Run from the ``autolens_workspace_test`` repo root with the standard cache overrides::

    NUMBA_CACHE_DIR=/tmp/numba_cache MPLCONFIGDIR=/tmp/matplotlib \\
        python scripts/cluster/visualization.py
"""

import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D

import autolens as al


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

 - ``data.fits`` for the cluster background image (``Array2D`` so ``.native`` returns a plain numpy
   2D array suitable for ``imshow``).
 - ``point_datasets.csv`` — one ``PointDataset`` per source, carrying that source's image-plane
   positions and redshift.
 - ``tracer.json`` — the true ``Tracer``; gives multi-plane critical curves and the cosmology /
   lens redshift used by the kpc scale bar.
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


"""
__Common Styling__

The Wong (2011) palette — the de facto standard for colour-blind safe qualitative encoding in
scientific figures (https://www.nature.com/articles/nmeth.1618) — is hard-coded so this script has
zero optional-dependency surface. With only two sources here we use the first two entries; more
sources just take the next colours in order.

``gnuplot2`` (perceptually-ordered black → red → yellow → white) plus a percentile-driven
``LogNorm`` is what lets a single image span the BCG core, intra-cluster light, and faint arcs
without saturating either end.
"""
WONG_PALETTE = [
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # bluish green
    "#F0E442",  # yellow
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
    "#000000",  # black
]
CMAP = "gnuplot2"


def make_lognorm(image: np.ndarray) -> LogNorm:
    """Return a ``LogNorm`` driven by the positive-pixel percentiles of *image*.

    Cluster imaging has bright cores and faint background; a fixed (vmin, vmax) does not transfer
    between simulators. The 5th-percentile of *positive* pixels gives a vmin that suppresses sky
    noise without clipping faint arcs; the 99.5th-percentile gives a vmax that keeps the BCG core
    from saturating. ``LogNorm`` requires strictly positive bounds, so non-positive pixels are
    discarded before computing percentiles.
    """
    positive = image[image > 0]
    vmin = float(np.percentile(positive, 5.0))
    vmax = float(np.percentile(positive, 99.5))
    return LogNorm(vmin=vmin, vmax=vmax)


def field_extent_from(image: np.ndarray, pixel_scale: float) -> tuple:
    """Return ``imshow`` extent ``(left, right, bottom, top)`` for an origin-centred field."""
    half_y = image.shape[0] * pixel_scale / 2.0
    half_x = image.shape[1] * pixel_scale / 2.0
    return (-half_x, half_x, -half_y, half_y)


def draw_lens_markers(ax, *, marker_size: int = 220) -> None:
    """Plot the BCG centres (white stars) and the host-halo centre (white plus) on *ax*.

    Conventional cluster-figure markers, drawn in white with a thin black edge so they read against
    both bright (BCG core) and dark (sky) backgrounds. In this baseline the BCG and halo coincide
    at the origin so the markers stack — they visibly separate as soon as a real fit displaces the
    inferred halo centre.
    """
    bcg_y, bcg_x = main_lens_centres.array[0]
    sat_y, sat_x = main_lens_centres.array[1]
    halo_y, halo_x = host_halo_centre.array[0]
    ax.scatter(
        [bcg_x],
        [bcg_y],
        marker="*",
        s=marker_size,
        c="white",
        edgecolors="black",
        linewidths=0.8,
        zorder=5,
        label="BCG / member centres",
    )
    ax.scatter(
        [sat_x],
        [sat_y],
        marker="*",
        s=marker_size * 0.6,
        c="white",
        edgecolors="black",
        linewidths=0.8,
        zorder=5,
    )
    ax.scatter(
        [halo_x],
        [halo_y],
        marker="P",
        s=marker_size * 0.7,
        c="white",
        edgecolors="black",
        linewidths=0.8,
        zorder=5,
        label="Host halo centre",
    )


def draw_kpc_scale_bar(
    ax, *, lens_redshift: float, cosmology, kpc: float = 50.0
) -> None:
    """Draw a *kpc* physical scale bar in the bottom-left of *ax*.

    50 kpc at ``z = 0.5`` is ~8" — visible against a 100" field without dominating the panel.
    """
    arcsec_per_kpc = float(cosmology.arcsec_per_kpc_proper(lens_redshift))
    bar_arcsec = kpc * arcsec_per_kpc
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    x0 = xlim[0] + 0.05 * (xlim[1] - xlim[0])
    y0 = ylim[0] + 0.05 * (ylim[1] - ylim[0])
    ax.plot([x0, x0 + bar_arcsec], [y0, y0], color="white", lw=3.0, zorder=5)
    ax.text(
        x0 + bar_arcsec / 2.0,
        y0 + 0.02 * (ylim[1] - ylim[0]),
        f"{int(kpc)} kpc",
        color="white",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
        zorder=5,
    )


image_native = np.asarray(data.native)
extent = field_extent_from(image_native, PIXEL_SCALE)


"""
__Paths__

Mirrors the imaging visualization integration test: clean a per-script ``images/`` directory on
each run so assertions reflect this run only.
"""
image_path = Path("scripts") / "cluster" / "images" / "visualization"

if image_path.exists():
    shutil.rmtree(image_path)
image_path.mkdir(parents=True)


"""
__Plot 1 — Overlaid Positions__

Why this view matters at cluster scale: the default per-source ``subplot_point_dataset`` puts each
source on its own axes, so multiple-image groups belonging to different sources can never be
compared spatially against the cluster image. Rendering the cluster image once with a LogNorm
stretch and overlaying every source's positions in a distinct colour fixes that — the legend
carries name *and* redshift so multi-plane lensing is visible at a glance.
"""
print("Running plot 1 — overlaid positions...")
_t0 = time.perf_counter()

fig, ax = plt.subplots(figsize=(8.5, 8.0))
ax.imshow(
    image_native,
    cmap=CMAP,
    norm=make_lognorm(image_native),
    extent=extent,
    origin="lower",
)
for i, dataset in enumerate(point_datasets):
    colour = WONG_PALETTE[i % len(WONG_PALETTE)]
    positions = np.asarray(dataset.positions.array)
    ax.scatter(
        positions[:, 1],
        positions[:, 0],
        marker="o",
        s=80,
        facecolors="none",
        edgecolors=colour,
        linewidths=1.6,
        zorder=4,
        label=f"{dataset.name}  (z = {dataset.redshift:.2f})",
    )
draw_lens_markers(ax)
draw_kpc_scale_bar(
    ax,
    lens_redshift=tracer.galaxies[0].redshift,
    cosmology=tracer.cosmology,
)
ax.set_xlabel('x (")')
ax.set_ylabel('y (")')
ax.set_title("Cluster image with overlaid multiple-image positions")
ax.legend(loc="upper right", framealpha=0.85, fontsize=9)
fig.tight_layout()
fig.savefig(image_path / "visualization_overlaid_positions.png", dpi=150)
plt.close(fig)

print(f"plot 1 complete in {time.perf_counter() - _t0:.2f}s")

assert (
    image_path / "visualization_overlaid_positions.png"
).exists(), "visualization_overlaid_positions.png missing"
assert (
    image_path / "visualization_overlaid_positions.png"
).stat().st_size > 0, "visualization_overlaid_positions.png is empty"
print("visualization_overlaid_positions.png OK")


"""
__Plot 2 — Per-Source Subplot Grid__

Why this view matters at cluster scale: with N sources at distinct redshifts in the same field,
inspecting each source independently is the most direct way to compare image-plane configurations
side by side. ``aplt`` has no equivalent today — there is no function that takes a
``List[PointDataset]`` plus the cluster image and produces a per-source zoom grid.

Bounding-box-driven cropping with a 30 %-of-span margin (5" floor) keeps every multiple image well
inside the panel without leaving the panel almost empty for compact groups.
"""
print("\nRunning plot 2 — per-source subplot grid...")
_t0 = time.perf_counter()

n_sources = len(point_datasets)
fig, axes = plt.subplots(1, n_sources, figsize=(6.0 * n_sources, 6.0), squeeze=False)
axes = axes[0]

lognorm = make_lognorm(image_native)

for i, (dataset, ax) in enumerate(zip(point_datasets, axes)):
    colour = WONG_PALETTE[i % len(WONG_PALETTE)]
    positions = np.asarray(dataset.positions.array)

    bbox_y = (positions[:, 0].min(), positions[:, 0].max())
    bbox_x = (positions[:, 1].min(), positions[:, 1].max())
    span_y = bbox_y[1] - bbox_y[0]
    span_x = bbox_x[1] - bbox_x[0]
    margin = max(0.30 * max(span_y, span_x), 5.0)

    ax.imshow(image_native, cmap=CMAP, norm=lognorm, extent=extent, origin="lower")
    ax.scatter(
        positions[:, 1],
        positions[:, 0],
        marker="o",
        s=120,
        facecolors="none",
        edgecolors=colour,
        linewidths=2.0,
        zorder=4,
    )
    draw_lens_markers(ax, marker_size=160)
    ax.set_xlim(bbox_x[0] - margin, bbox_x[1] + margin)
    ax.set_ylim(bbox_y[0] - margin, bbox_y[1] + margin)
    ax.set_xlabel('x (")')
    if i == 0:
        ax.set_ylabel('y (")')
    ax.set_title(f"{dataset.name}   z = {dataset.redshift:.2f}", color=colour)

fig.suptitle("Per-source zoom panels (each panel cropped to its image group)", y=1.02)
fig.tight_layout()
fig.savefig(
    image_path / "visualization_per_source_grid.png", dpi=150, bbox_inches="tight"
)
plt.close(fig)

print(f"plot 2 complete in {time.perf_counter() - _t0:.2f}s")

assert (
    image_path / "visualization_per_source_grid.png"
).exists(), "visualization_per_source_grid.png missing"
assert (
    image_path / "visualization_per_source_grid.png"
).stat().st_size > 0, "visualization_per_source_grid.png is empty"
print("visualization_per_source_grid.png OK")


"""
__Plot 3 — Critical-Curve Overlay__

Why this view matters at cluster scale: the tangential critical curve at cluster scale is *not*
one tidy ellipse — a massive host halo produces a large outer curve while individual member
galaxies produce smaller inner curves around their cores. Showing the full curve set tells the
modeller which sources sit in the strongly-lensed region and which sit just outside. The default
``aplt`` line weight was tuned for galaxy-scale Einstein radii (~1–2"); at 100" it draws a solid
white ribbon obliterating the underlying image.

We use ``LensCalc.from_tracer(tracer, use_multi_plane=True, plane_j=-1)`` to anchor the
deflection callable to the *furthest* source plane (z = 2 here), which is the deepest light cone in
the system and the convention adopted throughout PyAutoLens for multi-plane critical-curve work.
``cyan`` reads against ``gnuplot2`` (black → red → yellow → white); ``linewidth=0.8`` and
``alpha=0.7`` keep the curve informative rather than dominant.
"""
print("\nRunning plot 3 — critical-curve overlay...")
_t0 = time.perf_counter()

lens_calc = al.LensCalc.from_tracer(tracer, use_multi_plane=True, plane_j=-1)
# The cluster host's tangential critical curve sits at ~30-50"; the assertion below requires the
# grid to extend past it. Opt out of PYAUTO_SMALL_DATASETS' 15x15 @ 0.6" shrink for this one grid.
viz_grid = al.Grid2D.uniform(
    shape_native=(200, 200), pixel_scales=0.5, respect_small_datasets=False
)
tangential_curves = lens_calc.tangential_critical_curve_list_from(grid=viz_grid)

CURVE_COLOR = "#00FFFF"
CURVE_LINEWIDTH = 0.8
CURVE_ALPHA = 0.7

fig, ax = plt.subplots(figsize=(8.5, 8.0))
ax.imshow(
    image_native,
    cmap=CMAP,
    norm=make_lognorm(image_native),
    extent=extent,
    origin="lower",
)
for curve in tangential_curves:
    pts = np.asarray(curve.array)
    ax.plot(
        pts[:, 1],
        pts[:, 0],
        color=CURVE_COLOR,
        linewidth=CURVE_LINEWIDTH,
        alpha=CURVE_ALPHA,
        zorder=3,
    )
draw_lens_markers(ax)

curve_proxy = Line2D(
    [0],
    [0],
    color=CURVE_COLOR,
    linewidth=CURVE_LINEWIDTH,
    alpha=CURVE_ALPHA,
    label=(
        f"Tangential critical curve (multi-plane, "
        f"z_src = {max(g.redshift for g in tracer.galaxies):.2f})"
    ),
)
ax.legend(handles=[curve_proxy], loc="upper right", framealpha=0.85, fontsize=9)
ax.set_xlabel('x (")')
ax.set_ylabel('y (")')
ax.set_title("Cluster image with multi-plane tangential critical curve")
fig.tight_layout()
fig.savefig(image_path / "visualization_critical_curves.png", dpi=150)
plt.close(fig)

print(f"plot 3 complete in {time.perf_counter() - _t0:.2f}s")

assert (
    image_path / "visualization_critical_curves.png"
).exists(), "visualization_critical_curves.png missing"
assert (
    image_path / "visualization_critical_curves.png"
).stat().st_size > 0, "visualization_critical_curves.png is empty"
print("visualization_critical_curves.png OK")


"""
At least one tangential critical curve must have been recovered for a 10^15.3 M_sun host halo;
zero curves at this halo mass would indicate a regression in either the simulator (mass too low)
or the LensCalc multi-plane plumbing.
"""
assert (
    len(tangential_curves) > 0
), "no tangential critical curves recovered (expected at least one for a 10^15.3 M_sun host)"
print(f"tangential_curves recovered: {len(tangential_curves)} OK")


print("\nAll visualization assertions passed.")
