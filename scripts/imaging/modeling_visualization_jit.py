"""
End-to-end test: jit-cached visualization during a real Nautilus model-fit.
==========================================================================

Exercises the full Path A pipeline shipped across PyAutoArray #288, PyAutoLens
#445, and the PyAutoFit change that turns ``Analysis.fit_for_visualization``
into a lazily-cached ``jax.jit(self.fit_from)``.

This test runs in two parts:

Part 1 — **MGE caching probe.** Uses the same MGE parametric lens + MGE source
model as the offline PoC at ``scripts/jax_likelihood_functions/imaging/mge_pytree.py``.
Calls ``analysis.fit_for_visualization(instance)`` twice and asserts the
second call is much faster than the first (confirming the compiled function
is cached on the analysis instance, not recompiled per visualization).

Part 2 — **Live Nautilus quick-update with MGE linear profiles.** Runs a
real (short) Nautilus fit with an MGE lens (``GaussianGradient`` basis +
``NFWSph`` mass) and MGE source — both use linear light profiles whose
``intensity`` is solved by the inversion. With the ``pytree_token`` fix on
``LightProfileLinear``, the ``linear_light_profile_intensity_dict`` lookup
survives the JAX pytree round-trip and no ``KeyError`` is raised. Asserts
that ``subplot_fit.png`` files land on disk, proving the JIT-cached
fit_for_visualization fires correctly during the live search callback.

This script deliberately opts in with
``AnalysisImaging(use_jax=True, use_jax_for_visualization=True)``. Default
model-fit scripts elsewhere in the workspace leave both flags at ``False``
and are therefore untouched by this change.
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

Re-use the jax_test MGE dataset that the jax_likelihood_functions scripts rely
on. Auto-simulate if missing.
"""
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
)

mask_radius = 3.5
mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=mask_radius,
)
dataset = dataset.apply_mask(mask=mask)
dataset = dataset.apply_over_sampling(over_sample_size_lp=4)


"""
============================================================================
Part 1 — MGE caching probe
============================================================================

Model: MGE parametric lens (Basis of 20 Gaussians + NFWSph + ExternalShear)
and MGE parametric source. Mirrors ``scripts/jax_likelihood_functions/imaging/mge.py``
and the shipped offline PoC at ``mge_pytree.py``.
"""
print("\n" + "=" * 72)
print("Part 1: MGE caching probe")
print("=" * 72)

bulge_mge = al.model_util.mge_model_from(
    mask_radius=mask_radius, total_gaussians=20, centre_prior_is_uniform=True
)

mass_mge = af.Model(al.mp.NFWSph)

total_gaussians = 3
log10_sigma_list = np.linspace(-2, np.log10(mask_radius), total_gaussians)

centre_0 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
centre_1 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)

gaussian_list = af.Collection(
    af.Model(al.lmp_linear.GaussianGradient) for _ in range(total_gaussians)
)
for i, gaussian in enumerate(gaussian_list):
    gaussian.centre.centre_0 = centre_0
    gaussian.centre.centre_1 = centre_1
    gaussian.ell_comps = gaussian_list[0].ell_comps
    gaussian.sigma = 10 ** log10_sigma_list[i]
    gaussian.mass_to_light_ratio = 10.0
    gaussian.mass_to_light_gradient = 1.0

bulge_mge = af.Model(al.lp_basis.Basis, profile_list=list(gaussian_list))
shear_mge = af.Model(al.mp.ExternalShear)

lens_mge = af.Model(
    al.Galaxy, redshift=0.5, bulge=bulge_mge, mass=mass_mge, shear=shear_mge
)

source_bulge_mge = al.model_util.mge_model_from(
    mask_radius=mask_radius, total_gaussians=20, centre_prior_is_uniform=False
)
source_mge = af.Model(al.Galaxy, redshift=1.0, bulge=source_bulge_mge)

model_mge = af.Collection(galaxies=af.Collection(lens=lens_mge, source=source_mge))

register_model(model_mge)

analysis_mge = al.AnalysisImaging(
    dataset=dataset,
    use_jax=True,
    use_jax_for_visualization=True,
)

instance_mge = model_mge.instance_from_prior_medians()

t0 = time.perf_counter()
fit_1 = analysis_mge.fit_for_visualization(instance_mge)
jax.block_until_ready(fit_1.log_likelihood)
t1 = time.perf_counter()
compile_time = t1 - t0
print(f"First call (compile + run): {compile_time:.3f}s")
print(f"  log_likelihood leaf type: {type(fit_1.log_likelihood).__name__}")
assert isinstance(fit_1.log_likelihood, jnp.ndarray), (
    f"expected jax.Array, got {type(fit_1.log_likelihood)}"
)

t0 = time.perf_counter()
fit_2 = analysis_mge.fit_for_visualization(instance_mge)
jax.block_until_ready(fit_2.log_likelihood)
t1 = time.perf_counter()
cached_time = t1 - t0
print(f"Second call (cached):       {cached_time:.3f}s")
print(f"Speedup:                    {compile_time / max(cached_time, 1e-9):.1f}x")

assert cached_time < compile_time * 0.5, (
    f"Cached call ({cached_time:.3f}s) not faster than compile "
    f"({compile_time:.3f}s) — JIT cache is not being hit."
)
assert analysis_mge._jitted_fit_from is not None, (
    "expected _jitted_fit_from to be cached on the analysis instance after first call"
)
print("PASS: MGE jit-cached fit_for_visualization works and is reused.")


"""
============================================================================
Part 2 — Live Nautilus quick-update with MGE linear light profiles
============================================================================

Model: MGE parametric lens (Basis of GaussianGradient + NFWSph mass) and
MGE parametric source. Linear light profiles are used, so the
``linear_light_profile_intensity_dict`` lookup is exercised during
visualization. With the ``pytree_token`` fix on ``LightProfileLinear``,
dict lookups survive the JAX pytree round-trip and no ``KeyError`` is raised.

The live search fires quick-update visualization every
``iterations_per_quick_update`` calls; we verify subplot_fit.png lands on disk.
"""
print("\n" + "=" * 72)
print("Part 2: Live Nautilus with MGE linear profiles + jit-visualization")
print("=" * 72)

mass_mge2 = af.Model(al.mp.NFWSph)

total_gaussians2 = 3
log10_sigma_list2 = np.linspace(-2, np.log10(mask_radius), total_gaussians2)

centre_0_2 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
centre_1_2 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)

gaussian_list2 = af.Collection(
    af.Model(al.lmp_linear.GaussianGradient) for _ in range(total_gaussians2)
)
for i, gaussian in enumerate(gaussian_list2):
    gaussian.centre.centre_0 = centre_0_2
    gaussian.centre.centre_1 = centre_1_2
    gaussian.ell_comps = gaussian_list2[0].ell_comps
    gaussian.sigma = 10 ** log10_sigma_list2[i]
    gaussian.mass_to_light_ratio = 10.0
    gaussian.mass_to_light_gradient = 1.0

bulge_mge2 = af.Model(al.lp_basis.Basis, profile_list=list(gaussian_list2))

lens_mge2 = af.Model(
    al.Galaxy, redshift=0.5, bulge=bulge_mge2, mass=mass_mge2
)

source_bulge_mge2 = al.model_util.mge_model_from(
    mask_radius=mask_radius, total_gaussians=20, centre_prior_is_uniform=False
)
source_mge2 = af.Model(al.Galaxy, redshift=1.0, bulge=source_bulge_mge2)

model_mge2 = af.Collection(
    galaxies=af.Collection(lens=lens_mge2, source=source_mge2)
)

register_model(model_mge2)

analysis_mge2 = al.AnalysisImaging(
    dataset=dataset,
    use_jax=True,
    use_jax_for_visualization=True,
)

output_root = Path("scripts") / "imaging" / "images" / "modeling_visualization_jit"
if output_root.exists():
    shutil.rmtree(output_root)
output_root.mkdir(parents=True)

search = af.Nautilus(
    path_prefix=str(output_root),
    name="mge_linear",
    n_live=50,
    n_like_max=1500,
    iterations_per_quick_update=500,
    number_of_cores=1,
)

print("Running Nautilus ...")
result = search.fit(model=model_mge2, analysis=analysis_mge2)

# The Nautilus output goes to output/<path_prefix>/<name>/<hash>/image/
# The quick-update visualizer writes fit.png (via subplot_fit function)
# to that image folder during each quick update.
output_search_root = Path("output") / output_root / "mge_linear"
produced_pngs = list(output_search_root.rglob("fit.png"))
print(f"fit.png files produced: {len(produced_pngs)}")
for p in produced_pngs:
    print(f"  {p}")
assert len(produced_pngs) > 0, (
    f"no fit.png produced under {output_search_root} — "
    "quick-update visualization did not fire"
)
assert analysis_mge2._jitted_fit_from is not None, (
    "expected _jitted_fit_from to be cached on the analysis instance during search"
)

print(
    "\nPASS: jit-cached fit_for_visualization fires during Nautilus quick updates "
    "with MGE linear profiles, fit.png written, no KeyError from "
    "linear_light_profile_intensity_dict lookup."
)
