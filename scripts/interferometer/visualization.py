"""
Visualization: Interferometer Analysis
=======================================

Tests that `AnalysisInterferometer.visualize_before_fit` and `visualize` output all expected files
to disk and that each output has the correct FITS HDU structure.

Dataset: MGE lens bulge + PowerLaw mass on the build interferometer dataset (no_lens_light).

Structure
---------
1. `visualize_before_fit` runs once with a parametric source (fastest) and writes all
   before-fit outputs (dataset.png/.fits, image_with_positions.png,
   adapt_images.png, adapt_images.fits) to the main `visualization/` folder.

2. `visualize` runs once per source type, each writing into its own subfolder:
     visualization/parametric/   — Sersic light-profile source
     visualization/rectangular/  — RectangularAdaptImage pixelization
     visualization/delaunay/     — Delaunay pixelization

   Each subfolder contains only the source-dependent comparison plots:
     fit.png, tracer.png, fit_dirty_images.png, fit_real_space.png  (all three sources)
     inversion_0_0.png                                               (rectangular and delaunay only)

   A minimal `config_source/visualize/plots.yaml` (pushed before these runs) limits
   output to just those files so the per-source runs stay fast.

Expected outputs are derived directly from the source code of:
  - autolens/interferometer/model/visualizer.py         (VisualizerInterferometer)
  - autolens/interferometer/model/plotter.py            (PlotterInterferometer)
  - autogalaxy/interferometer/plot/fit_interferometer_plots.py (fits_galaxy_images, fits_dirty_images)
  - autolens/analysis/plotter.py                        (Plotter: tracer, galaxies, inversion)
  - autogalaxy/analysis/plotter.py                      (Plotter: galaxies, inversion)
"""

import shutil
from os import path
from pathlib import Path
from types import SimpleNamespace

# Push the bespoke all-true plots.yaml before any visualization method reads config.
from autoconf import conf

conf.instance.push(
    new_path=path.join(path.dirname(path.realpath(__file__)), "config"),
    output_path=path.join(path.dirname(path.realpath(__file__)), "images"),
)

from astropy.io import fits as astropy_fits

import autofit as af
import autolens as al
from autolens.interferometer.model.visualizer import VisualizerInterferometer


"""
__Dataset__

Build interferometer with_lens_light: data.fits, noise_map.fits, uv_wavelengths.fits, positions.json.
real_space_mask matches the settings used in scripts/interferometer/model_fit.py.
TransformerDFT is used (dataset is small enough for exact DFT).
"""

dataset_path = path.join("dataset", "build", "interferometer", "no_lens_light")

"""
__Dataset Auto-Simulation__

If the dataset does not already exist on your system, it will be created by running the corresponding
simulator script. This ensures that all example scripts can be run without manually simulating data first.
"""
if not path.exists(dataset_path):
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "scripts/interferometer/simulator/no_lens_light.py"],
        check=True,
    )

mask_radius = 3.0

real_space_mask = al.Mask2D.circular(
    shape_native=(100, 100),
    pixel_scales=0.2,
    radius=mask_radius,
)

dataset = al.Interferometer.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    uv_wavelengths_path=path.join(dataset_path, "uv_wavelengths.fits"),
    real_space_mask=real_space_mask,
    transformer_class=al.TransformerDFT,
)


"""
__Positions__

Loaded from positions.json; used to trigger image_with_positions visualization.
"""

positions = al.Grid2DIrregular(
    al.from_json(file_path=path.join(dataset_path, "positions.json"))
)
positions_likelihood = al.PositionsLH(positions=positions, threshold=1.0)


"""
__Lens Model__

Lens: PowerLaw mass (no lens light) — matches the simulator in
scripts/interferometer/simulator/no_lens_light.py.
"""
mass = af.Model(al.mp.PowerLaw)
mass.centre.centre_0 = 0.0
mass.centre.centre_1 = 0.0
mass.ell_comps.ell_comps_0 = 0.0
mass.ell_comps.ell_comps_1 = 0.0
mass.einstein_radius = 1.6
mass.slope = 1.8

lens = af.Model(al.Galaxy, redshift=0.5, mass=mass)


"""
__Image Plane Mesh Grid__

Used by both Delaunay and RectangularAdaptImage pixelizations.
"""
image_mesh = al.image_mesh.Overlay(shape=(26, 26))

image_plane_mesh_grid = image_mesh.image_plane_mesh_grid_from(
    mask=real_space_mask,
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

galaxy_name_image_dict provides per-galaxy images used by adaptive regularization.
galaxy_name_image_plane_mesh_grid_dict provides the Overlay mesh grid for Delaunay
and RectangularAdaptImage pixelizations.
dirty_image is the interferometer's real-space image equivalent.
"""

adapt_images = al.AdaptImages(
    galaxy_name_image_dict={
        "('galaxies', 'lens')": dataset.dirty_image,
        "('galaxies', 'source')": dataset.dirty_image,
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

analysis = al.AnalysisInterferometer(
    dataset=dataset,
    positions_likelihood_list=[positions_likelihood],
    adapt_images=adapt_images,
    use_jax=False,
    title_prefix="TEST"
)


"""
__Paths__

Minimal paths stub: VisualizerInterferometer only needs image_path and output_path.
Clean the output directory on each run so assertions reflect this run only.
"""

image_path = Path("scripts") / "interferometer" / "images" / "visualization"

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

Calls PlotterInterferometer.interferometer() -> dataset.png, dataset.fits
      Plotter.image_with_positions()         -> image_with_positions.png
      Plotter.adapt_images()                 -> adapt_images.png, adapt_images.fits
"""

print("Running visualize_before_fit (parametric source)...")

VisualizerInterferometer.visualize_before_fit(
    analysis=analysis,
    paths=paths,
    model=model_parametric,
)

print("visualize_before_fit complete.")

"""
__Assertions: visualize_before_fit__
"""

# ---- dataset.png / dataset.fits ----
# Source: PlotterInterferometer.interferometer() -> hdu_list_for_output_from with ext_name_list:
#   ["mask", "data", "noise_map", "uv_wavelengths"]
# HDU 0 is PrimaryHDU (mask), HDUs 1-3 are ImageHDU.

assert (image_path / "dataset.png").exists(), "dataset.png missing"
print("dataset.png OK")

with astropy_fits.open(image_path / "dataset.fits") as hdul:
    assert len(hdul) == 4, f"dataset.fits: expected 4 HDUs, got {len(hdul)}"
    assert hdul[0].name == "MASK"
    assert hdul[1].name == "DATA"
    assert hdul[2].name == "NOISE_MAP"
    assert hdul[3].name == "UV_WAVELENGTHS"
print("dataset.fits OK")

# ---- image_with_positions.png ----
# Source: Plotter.image_with_positions() -> uses dataset.dirty_image as base

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
Only subplot_fit.png and subplot_tracer.png are generated for all three;
rectangular and delaunay also produce subplot_inversion_0.png.

Calls (governed by config_source/visualize/plots.yaml):
  fit.subplot_fit                 -> fit.png
  tracer.subplot_tracer           -> tracer.png
  fit_interferometer.subplot_fit_dirty_images -> fit_dirty_images.png
  fit_interferometer.subplot_fit_real_space   -> fit_real_space.png
  inversion.subplot_inversion     -> inversion_0.png  (pixelized sources only)
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

    VisualizerInterferometer.visualize(
        analysis=analysis,
        paths=sub_paths,
        instance=instance,
        during_analysis=False,
    )

    print(f"  visualize complete for {source_name}.")

    assert (sub_path / "fit.png").exists(), f"{source_name}/fit.png missing"
    print(f"  {source_name}/fit.png OK")
    assert (sub_path / "tracer.png").exists(), f"{source_name}/tracer.png missing"
    print(f"  {source_name}/tracer.png OK")
    assert (
        sub_path / "fit_dirty_images.png"
    ).exists(), f"{source_name}/fit_dirty_images.png missing"
    print(f"  {source_name}/fit_dirty_images.png OK")
    assert (
        sub_path / "fit_real_space.png"
    ).exists(), f"{source_name}/fit_real_space.png missing"
    print(f"  {source_name}/fit_real_space.png OK")
    if has_inversion:
        assert (
            sub_path / "inversion_0_0.png"
        ).exists(), f"{source_name}/inversion_0_0.png missing"
        print(f"  {source_name}/inversion_0_0.png OK")


print("All visualization assertions passed.")
