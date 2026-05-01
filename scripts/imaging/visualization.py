"""
Visualization: Imaging Analysis
================================

Tests that `AnalysisImaging.visualize_before_fit` and `visualize` output all expected files to disk
and that each output has the correct FITS HDU structure.

Dataset: MGE lens bulge + PowerLaw mass on build imaging (with_lens_light).

Structure
---------
1. `visualize_before_fit` runs once with a parametric source (fastest) and writes all
   before-fit outputs (dataset.png/.fits, image_with_positions.png, adapt_images.png/.fits)
   to the main `visualization/` folder.

2. `visualize` runs once per source type, each writing into its own subfolder:
     visualization/parametric/   — Sersic light-profile source
     visualization/rectangular/  — RectangularAdaptImage pixelization
     visualization/delaunay/     — Delaunay pixelization

   Each subfolder contains only the source-dependent comparison plots:
     fit.png, tracer.png (all three sources)
     inversion_0_0.png   (rectangular and delaunay only)

   A minimal `config_source/visualize/plots.yaml` (pushed before these runs) limits
   output to just those files so the per-source runs stay fast.

Expected outputs are derived directly from the source code of:
  - autolens/imaging/model/visualizer.py    (VisualizerImaging)
  - autolens/imaging/model/plotter.py       (PlotterImaging)
  - autolens/analysis/plotter.py            (Plotter: tracer, galaxies, inversion)
  - autogalaxy/analysis/plotter.py          (Plotter: galaxies, inversion)
  - autogalaxy/imaging/plot/fit_imaging_plots.py
"""

import shutil
import time
from os import path
from pathlib import Path
from types import SimpleNamespace

# Push the bespoke all-true plots.yaml before any visualization method reads config.
# This must come before autolens imports trigger config reads in visualization code paths.
from autoconf import conf

conf.instance.push(
    new_path=path.join(path.dirname(path.realpath(__file__)), "config"),
    output_path=path.join(path.dirname(path.realpath(__file__)), "images"),
)

import numpy as np
from astropy.io import fits as astropy_fits

import autofit as af
import autolens as al
from autolens.imaging.model.visualizer import VisualizerImaging


"""
__Dataset__

Build imaging (with_lens_light): pixel_scale=0.2", 80x80 grid.
Uses fixed over_sample_size to avoid loading snr_no_lens.fits.
"""

pixel_scale = 0.2

dataset_path = path.join("dataset", "build", "imaging", "with_lens_light")

"""
__Dataset Auto-Simulation__

If the dataset does not already exist on your system, it will be created by running the corresponding
simulator script. This ensures that all example scripts can be run without manually simulating data first.
"""
if not path.exists(dataset_path):
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "scripts/imaging/simulator/with_lens_light.py"],
        check=True,
    )

dataset = al.Imaging.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    pixel_scales=pixel_scale,
    over_sample_size_lp=2,
    over_sample_size_pixelization=2,
)

mask_radius = 3.0

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=mask_radius,
)

dataset = dataset.apply_mask(mask=mask)


"""
__Positions__

Two lensed image positions used to trigger image_with_positions visualization.
"""
positions = al.Grid2DIrregular([(-0.5, 1.0), (0.5, -1.0)])
positions_likelihood = al.PositionsLH(positions=positions, threshold=1.0)


"""
__Lens Model__

Lens: DevVaucouleursSph bulge + PowerLaw mass — matches the simulator in
scripts/imaging/simulator/with_lens_light.py.
"""
bulge = af.Model(al.lp.DevVaucouleursSph)
bulge.centre.centre_0 = 0.0
bulge.centre.centre_1 = 0.0
bulge.intensity = 0.1
bulge.effective_radius = 0.8

mass = af.Model(al.mp.PowerLaw)
mass.centre.centre_0 = 0.0
mass.centre.centre_1 = 0.0
mass.ell_comps.ell_comps_0 = 0.05
mass.ell_comps.ell_comps_1 = 0.1
mass.einstein_radius = 1.6
mass.slope = 2.0

lens = af.Model(al.Galaxy, redshift=0.5, bulge=bulge, mass=mass)


"""
__Image Plane Mesh Grid__

Used by both Delaunay and RectangularAdaptImage pixelizations.
"""
image_mesh = al.image_mesh.Overlay(shape=(26, 26))

image_plane_mesh_grid = image_mesh.image_plane_mesh_grid_from(
    mask=dataset.mask,
)


"""
__Source Models__

Three source configurations share the same lens; only the source galaxy differs.
"""

# --- Parametric (ExponentialSph — matches simulator source) ---
source_bulge = af.Model(al.lp.ExponentialSph)
source_bulge.centre.centre_0 = 0.5
source_bulge.centre.centre_1 = 0.25
source_bulge.intensity = 0.3
source_bulge.effective_radius = 0.1
source_parametric = af.Model(al.Galaxy, redshift=1.0, bulge=source_bulge)
model_parametric = af.Collection(
    galaxies=af.Collection(lens=lens, source=source_parametric)
)

# --- Rectangular pixelization ---
mesh_rect = al.mesh.RectangularAdaptImage(shape=(22, 22))
reg_rect = al.reg.Constant(coefficient=1.0)
pix_rect = al.Pixelization(mesh=mesh_rect, regularization=reg_rect)
source_rectangular = af.Model(al.Galaxy, redshift=1.0, pixelization=pix_rect)
model_rectangular = af.Collection(
    galaxies=af.Collection(lens=lens, source=source_rectangular)
)

# --- Delaunay pixelization ---
mesh_del = al.mesh.Delaunay(pixels=image_plane_mesh_grid.shape[0], zeroed_pixels=0)
reg_del = al.reg.ConstantSplit(coefficient=1.0)
pix_del = al.Pixelization(mesh=mesh_del, regularization=reg_del)
source_delaunay = af.Model(al.Galaxy, redshift=1.0, pixelization=pix_del)
model_delaunay = af.Collection(
    galaxies=af.Collection(lens=lens, source=source_delaunay)
)


"""
__Adapt Images__

galaxy_name_image_dict is used to test for adapt_images.png, but is not actually
used in this likelihood function.
"""
adapt_images = al.AdaptImages(
    galaxy_name_image_dict={
        "('galaxies', 'lens')": dataset.data,
        "('galaxies', 'source')": dataset.data,
    },
    galaxy_name_image_plane_mesh_grid_dict={
        "('galaxies', 'source')": image_plane_mesh_grid
    },
)


"""
__Analysis__

A single analysis object is shared across all three source runs: it holds the
dataset only; the source type is determined by the instance passed to visualize.
"""
analysis = al.AnalysisImaging(
    dataset=dataset,
    positions_likelihood_list=[positions_likelihood],
    adapt_images=adapt_images,
    use_jax=True,
    title_prefix="TEST",
)


"""
__Paths__

Minimal paths stub: VisualizerImaging only needs image_path and output_path.
Clean the output directory on each run so assertions reflect this run only.
"""

image_path = Path("scripts") / "imaging" / "images" / "visualization"

if image_path.exists():
    shutil.rmtree(image_path)

image_path.mkdir(parents=True)

output_path = image_path / "output"
output_path.mkdir(parents=True)

paths = SimpleNamespace(
    image_path=image_path,
    output_path=output_path,
)


"""
__Visualize Before Fit__

Uses the parametric source (fastest) for all before-fit outputs.

Calls PlotterImaging.imaging()          -> dataset.png, dataset.fits
      Plotter.image_with_positions()    -> image_with_positions.png
      Plotter.adapt_images()            -> adapt_images.png, adapt_images.fits
"""

print("Running visualize_before_fit (parametric source)...")

_t0 = time.perf_counter()
VisualizerImaging.visualize_before_fit(
    analysis=analysis,
    paths=paths,
    model=model_parametric,
)
print(f"visualize_before_fit complete in {time.perf_counter() - _t0:.2f}s")

"""
__Assertions: visualize_before_fit__
"""

# ---- dataset.fits ----
# Source: PlotterImaging.imaging() -> hdu_list_for_output_from with ext_name_list:
#   ["mask", "data", "noise_map", "psf", "over_sample_size_lp", "over_sample_size_pixelization"]
# HDU 0 is PrimaryHDU (first value), HDUs 1-5 are ImageHDU.

assert (image_path / "dataset.png").exists(), "dataset.png missing"
print("dataset.png OK")

with astropy_fits.open(image_path / "dataset.fits") as hdul:
    assert len(hdul) == 6, f"dataset.fits: expected 6 HDUs, got {len(hdul)}"
    assert hdul[0].name == "MASK"
    assert hdul[1].name == "DATA"
    assert hdul[2].name == "NOISE_MAP"
    assert hdul[3].name == "PSF"
    assert hdul[4].name == "OVER_SAMPLE_SIZE_LP"
    assert hdul[5].name == "OVER_SAMPLE_SIZE_PIXELIZATION"
    assert hdul[1].data.ndim == 2, "DATA HDU should be 2D"
print("dataset.fits OK")

# ---- image_with_positions.png ----
# Source: Plotter.image_with_positions() -> image_plotter.set_filename("image_with_positions")

assert (
    image_path / "image_with_positions.png"
).exists(), "image_with_positions.png missing"
print("image_with_positions.png OK")

# ---- adapt_images.fits ----
# Source: Plotter.adapt_images() -> hdu_list_for_output_from with ext_name_list:
#   ["mask", "('galaxies', 'lens')", "('galaxies', 'source')"]
# HDU 0 = MASK (Primary), HDU 1 = lens key (uppercased), HDU 2 = source key (uppercased).

assert (image_path / "adapt_images.png").exists(), "adapt_images.png missing"
print("adapt_images.png OK")

with astropy_fits.open(image_path / "adapt_images.fits") as hdul:
    assert len(hdul) == 3, f"adapt_images.fits: expected 3 HDUs, got {len(hdul)}"
    assert hdul[0].name == "MASK"
print("adapt_images.fits OK")


"""
__Push Minimal Config for Per-Source Runs__

Override the all-true config with a minimal one that only enables:
  fit.subplot_fit, tracer.subplot_tracer, inversion.subplot_inversion.
All other toggles are explicitly set to false so no extra files are written.
"""
conf.instance.push(
    new_path=path.join(path.dirname(path.realpath(__file__)), "config_source"),
    output_path=path.join(path.dirname(path.realpath(__file__)), "images"),
)


"""
__Per-Source Visualization__

For each source type, visualize is run in a dedicated subfolder.
Only fit.png and tracer.png are generated for all three; rectangular and delaunay
also produce inversion_0_0.png.

Calls (governed by config_source/visualize/plots.yaml):
  fit.subplot_fit       -> fit.png
  tracer.subplot_tracer -> tracer.png
  inversion.subplot_inversion -> inversion_0_0.png  (pixelized sources only)
"""

source_runs = [
    ("parametric", model_parametric, False),
    ("rectangular", model_rectangular, True),
    ("delaunay", model_delaunay, True),
]

for source_name, model, has_inversion in source_runs:
    print(f"\nRunning visualize for source: {source_name}...")

    sub_path = image_path / source_name
    sub_path.mkdir(parents=True)
    sub_output = sub_path / "output"
    sub_output.mkdir(parents=True)
    sub_paths = SimpleNamespace(image_path=sub_path, output_path=sub_output)

    instance = model.instance_from_prior_medians()

    _t0 = time.perf_counter()
    VisualizerImaging.visualize(
        analysis=analysis,
        paths=sub_paths,
        instance=instance,
        during_analysis=False,
    )
    print(f"  visualize complete for {source_name} in {time.perf_counter() - _t0:.2f}s")

    assert (sub_path / "fit.png").exists(), f"{source_name}/fit.png missing"
    print(f"  {source_name}/fit.png OK")
    assert (sub_path / "tracer.png").exists(), f"{source_name}/tracer.png missing"
    print(f"  {source_name}/tracer.png OK")
    if has_inversion:
        assert (
            sub_path / "inversion_0_0.png"
        ).exists(), f"{source_name}/inversion_0_0.png missing"
        print(f"  {source_name}/inversion_0_0.png OK")


"""
__RGB Visualization__

Tests that `plot_array` correctly handles `Array2DRGB` inputs: no colormap,
no norm, no colorbar — the image is rendered via plain `imshow` as an RGB image.
"""

print("\nRunning RGB visualization test...")

import autolens.plot as aplt

rgb_values = np.stack(
    [dataset.data.native, dataset.data.native, dataset.data.native], axis=-1
)
rgb_values = np.clip(rgb_values, 0, None)

rgb_values_uint8 = (
    (rgb_values / rgb_values.max() * 255).astype(np.uint8)
    if rgb_values.max() > 0
    else np.zeros_like(rgb_values, dtype=np.uint8)
)

rgb_array = al.Array2DRGB(values=rgb_values_uint8, mask=dataset.mask)

aplt.plot_array(
    array=rgb_array,
    title="RGB Test",
    output_path=image_path,
    output_filename="rgb_array",
    output_format="png",
)

assert (image_path / "rgb_array.png").exists(), "rgb_array.png missing"
print("rgb_array.png OK")


print("All visualization assertions passed.")
