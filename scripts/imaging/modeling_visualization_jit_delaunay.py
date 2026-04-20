"""
End-to-end test: jit-cached visualization during a real Nautilus model-fit
with a DELAUNAY-pixelization source.
==========================================================================

Sibling to ``modeling_visualization_jit.py`` (MGE) and
``modeling_visualization_jit_rectangular.py`` (rectangular mesh). Exercises
the same Path A pipeline — ``Analysis.fit_for_visualization`` lazily-cached
as ``jax.jit(self.fit_from)`` — but with a Delaunay-triangulated source
whose centres are placed by a Hilbert image mesh.

This test runs in two parts:

Part 1 — **Caching probe.** Builds a parametric PowerLaw + shear lens with
a Delaunay source, calls ``analysis.fit_for_visualization`` twice and
asserts the second call is much faster (JIT cache hit).

Part 2 — **Live Nautilus quick-update.** Runs a real (short) Nautilus
fit. The visualizer callback fires during quick-updates and should produce
``fit.png`` on disk without error. The single-pixelized-source model keeps
the narrow fallback at
``PyAutoGalaxy/autogalaxy/galaxy/to_inversion.py:420-446`` viable for the
``galaxy_image_plane_mesh_grid_dict`` lookup that Delaunay requires.

Delaunay is the slowest of the three per-likelihood because of the 750
source pixels + Hilbert image_mesh; sampler budget is lower accordingly.
"""

import shutil
import time
from os import path
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import autofit as af
import autolens as al
from autofit.jax.pytrees import enable_pytrees, register_model

enable_pytrees()


"""
__Dataset__

Re-use the jax_test dataset. Auto-simulate if missing.
"""
sub_size = 4

dataset_path = path.join("dataset", "imaging", "jax_test")

if al.util.dataset.should_simulate(dataset_path):
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "scripts/jax_likelihood_functions/imaging/simulator.py"],
        check=True,
    )

dataset = al.Imaging.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    pixel_scales=0.2,
    over_sample_size_lp=sub_size,
    over_sample_size_pixelization=sub_size,
)

mask_radius = 2.6
mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=mask_radius,
)
dataset = dataset.apply_mask(mask=mask)

over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=dataset.grid,
    sub_size_list=[4, 2, 1],
    radial_list=[0.3, 0.6],
    centre_list=[(0.0, 0.0)],
)

dataset = dataset.apply_over_sampling(
    over_sample_size_lp=over_sample_size,
    over_sample_size_pixelization=1,
)


"""
__Mesh preloads__

Delaunay needs a precomputed image-plane mesh grid (Hilbert sampling +
circle-edge zeroing). ``pixels`` and ``edge_pixels_total`` are static JAX
shapes — hardcode per the reference at
``jax_likelihood_functions/imaging/delaunay.py``.
"""
pixels = 400
edge_pixels_total = 20

galaxy_name_image_dict = {
    "('galaxies', 'lens')": dataset.data,
    "('galaxies', 'source')": dataset.data,
}

image_mesh = al.image_mesh.Hilbert(pixels=pixels, weight_power=3.5, weight_floor=0.01)

image_plane_mesh_grid = image_mesh.image_plane_mesh_grid_from(
    mask=dataset.mask, adapt_data=galaxy_name_image_dict["('galaxies', 'source')"]
)

image_plane_mesh_grid = al.image_mesh.append_with_circle_edge_points(
    image_plane_mesh_grid=image_plane_mesh_grid,
    centre=mask.mask_centre,
    radius=mask_radius + mask.pixel_scale / 2.0,
    n_points=edge_pixels_total,
)

adapt_images = al.AdaptImages(
    galaxy_name_image_dict=galaxy_name_image_dict,
    galaxy_name_image_plane_mesh_grid_dict={
        "('galaxies', 'source')": image_plane_mesh_grid
    },
)


"""
__Model__

PowerLaw + ExternalShear lens, Delaunay-mesh source with AdaptSplit
regularization. Single pixelized source.
"""
mass = af.Model(al.mp.PowerLaw)
mass.centre.centre_0 = af.UniformPrior(lower_limit=0.2, upper_limit=0.4)
mass.centre.centre_1 = af.UniformPrior(lower_limit=-0.4, upper_limit=-0.2)
mass.einstein_radius = af.UniformPrior(lower_limit=1.5, upper_limit=1.7)
mass.ell_comps.ell_comps_0 = af.UniformPrior(
    lower_limit=0.11111111111111108, upper_limit=0.1111111111111111
)
mass.ell_comps.ell_comps_1 = af.UniformPrior(lower_limit=-0.01, upper_limit=0.01)

shear = af.Model(al.mp.ExternalShear)
shear.gamma_1 = af.UniformPrior(lower_limit=-0.001, upper_limit=0.001)
shear.gamma_2 = af.UniformPrior(lower_limit=-0.001, upper_limit=0.001)

lens = af.Model(al.Galaxy, redshift=0.5, mass=mass, shear=shear)

regularization = al.reg.AdaptSplit()
pixelization = af.Model(
    al.Pixelization,
    mesh=al.mesh.Delaunay(pixels=pixels, zeroed_pixels=edge_pixels_total),
    regularization=regularization,
)

source = af.Model(al.Galaxy, redshift=1.0, pixelization=pixelization)

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

register_model(model)


"""
============================================================================
Part 1 — Delaunay caching probe
============================================================================
"""
print("\n" + "=" * 72)
print("Part 1: Delaunay caching probe")
print("=" * 72)

analysis_probe = al.AnalysisImaging(
    dataset=dataset,
    adapt_images=adapt_images,
    raise_inversion_positions_likelihood_exception=False,
    use_jax=True,
    use_jax_for_visualization=True,
)

instance_probe = model.instance_from_prior_medians()

t0 = time.perf_counter()
fit_1 = analysis_probe.fit_for_visualization(instance_probe)
jax.block_until_ready(fit_1.log_likelihood)
t1 = time.perf_counter()
compile_time = t1 - t0
print(f"First call (compile + run): {compile_time:.3f}s")
print(f"  log_likelihood leaf type: {type(fit_1.log_likelihood).__name__}")
assert isinstance(fit_1.log_likelihood, jnp.ndarray), (
    f"expected jax.Array, got {type(fit_1.log_likelihood)}"
)

t0 = time.perf_counter()
fit_2 = analysis_probe.fit_for_visualization(instance_probe)
jax.block_until_ready(fit_2.log_likelihood)
t1 = time.perf_counter()
cached_time = t1 - t0
print(f"Second call (cached):       {cached_time:.3f}s")
print(f"Speedup:                    {compile_time / max(cached_time, 1e-9):.1f}x")

assert cached_time < compile_time * 0.5, (
    f"Cached call ({cached_time:.3f}s) not faster than compile "
    f"({compile_time:.3f}s) — JIT cache is not being hit."
)
assert analysis_probe._jitted_fit_from is not None, (
    "expected _jitted_fit_from to be cached on the analysis instance after first call"
)
print("PASS: Delaunay jit-cached fit_for_visualization works and is reused.")


"""
============================================================================
Part 2 — Live Nautilus quick-update with Delaunay pixelization
============================================================================

The live search fires quick-update visualization every
``iterations_per_quick_update`` calls; we verify ``fit.png`` lands on disk.
``n_like_max`` and ``iterations_per_quick_update`` are lower than the MGE /
rectangular scripts because Delaunay + 750 source pixels is the slowest
per-likelihood of the three.
"""
print("\n" + "=" * 72)
print("Part 2: Live Nautilus with Delaunay pixelization + jit-visualization")
print("=" * 72)

analysis_live = al.AnalysisImaging(
    dataset=dataset,
    adapt_images=adapt_images,
    raise_inversion_positions_likelihood_exception=False,
    use_jax=True,
    use_jax_for_visualization=True,
)

output_root = (
    Path("scripts") / "imaging" / "images" / "modeling_visualization_jit_delaunay"
)
if output_root.exists():
    shutil.rmtree(output_root)
output_root.mkdir(parents=True)

search = af.Nautilus(
    path_prefix=str(output_root),
    name="delaunay",
    n_live=50,
    n_like_max=500,
    iterations_per_quick_update=200,
    n_batch=10,
    number_of_cores=1,
)

print("Running Nautilus ...")
result = search.fit(model=model, analysis=analysis_live)

output_search_root = Path("output") / output_root / "delaunay"
produced_pngs = list(output_search_root.rglob("fit.png"))
print(f"fit.png files produced: {len(produced_pngs)}")
for p in produced_pngs:
    print(f"  {p}")
assert len(produced_pngs) > 0, (
    f"no fit.png produced under {output_search_root} — "
    "quick-update visualization did not fire"
)
assert analysis_live._jitted_fit_from is not None, (
    "expected _jitted_fit_from to be cached on the analysis instance during search"
)

print(
    "\nPASS: jit-cached fit_for_visualization fires during Nautilus quick updates "
    "with a Delaunay-pixelization source, fit.png written."
)
