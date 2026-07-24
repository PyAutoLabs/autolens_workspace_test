"""
End-to-end test: jit-cached visualization during a real Nautilus model-fit.
==========================================================================

Exercises the full Path A pipeline shipped across PyAutoArray #288, PyAutoLens
#445, and the PyAutoFit change that turns ``Analysis.fit_for_visualization``
into a lazily-cached ``jax.jit(self.fit_from)``.

This test runs in two parts:

Part 1 — **MGE caching probe.** Uses an MGE linear lens (GaussianGradient basis
+ NFWSph mass + ExternalShear) and MGE parametric source model. Calls
``analysis.fit_for_visualization(instance)`` twice and asserts the second call
is much faster than the first (confirming the compiled function is cached on the
analysis instance, not recompiled per visualization).

Part 2 — **Live Nautilus quick-update with MGE linear profiles.** Runs a real
(short) Nautilus fit with an MGE lens (``GaussianGradient`` basis + ``NFWSph``
mass) and MGE source — both use linear light profiles whose ``intensity`` is
solved by the inversion. With the ``pytree_token`` fix on
``LightProfileLinear``, the ``linear_light_profile_intensity_dict`` lookup
survives the JAX pytree round-trip and no ``KeyError`` is raised. Asserts that
``fit.png`` files land on disk, proving the JIT-cached fit_for_visualization
fires correctly during the live search callback.

This script deliberately opts in with
``AnalysisInterferometer(use_jax=True)``. Default model-fit scripts elsewhere
in the workspace leave the flag at ``False`` and are therefore untouched by
this change.

__Env__

Test-harness configuration (PyAutoHands docs/env_profile_redesign.md §10).
Interferometer JIT visualization path: real search, JAX, full-resolution
mask and real savefig.

ENV: real_output
"""

import shutil
import time
from os import path
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from autolens import with_test_mode_segment

import autofit as af
import autolens as al


"""
__Dataset__

Re-use the ``simple`` interferometer dataset. Auto-simulate if missing.
"""
mask_radius = 3.5

real_space_mask = al.Mask2D.circular(
    shape_native=(256, 256),
    pixel_scales=0.1,
    radius=mask_radius,
)

dataset_path = path.join("dataset", "interferometer", "simple")

if al.util.dataset.should_simulate(dataset_path):
    import subprocess
    import sys

    subprocess.run(
        [
            sys.executable,
            "scripts/interferometer/simulator/simple.py",
        ],
        check=True,
    )

dataset = al.Interferometer.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    uv_wavelengths_path=path.join(dataset_path, "uv_wavelengths.fits"),
    real_space_mask=real_space_mask,
    transformer_class=al.TransformerDFT,
)

positions = al.Grid2DIrregular(
    al.from_json(file_path=path.join(dataset_path, "positions.json"))
)


"""
============================================================================
Part 1 — MGE caching probe
============================================================================

Model: MGE linear lens (Basis of GaussianGradient + NFWSph mass + ExternalShear)
and MGE parametric source. Mirrors the linear MGE pattern from the imaging
analogue at ``scripts/imaging/modeling_visualization_jit.py``.
"""
print("\n" + "=" * 72)
print("Part 1: MGE caching probe")
print("=" * 72)

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


analysis_mge = al.AnalysisInterferometer(
    dataset=dataset,
    positions_likelihood_list=[al.PositionsLH(threshold=0.4, positions=positions)],
    use_jax=True,
)

instance_mge = model_mge.instance_from_prior_medians()

t0 = time.perf_counter()
fit_1 = analysis_mge.fit_for_visualization(instance_mge)
jax.block_until_ready(fit_1.log_likelihood)
t1 = time.perf_counter()
compile_time = t1 - t0
print(f"First call (compile + run): {compile_time:.3f}s")
print(f"  log_likelihood leaf type: {type(fit_1.log_likelihood).__name__}")
assert isinstance(
    fit_1.log_likelihood, jnp.ndarray
), f"expected jax.Array, got {type(fit_1.log_likelihood)}"

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
print("PASS: MGE jit-cached fit_for_visualization works and is reused.")


"""
__Visualization Sanity__

Phase D.1 rollout. Guards both the lensing-side regression class (silent
zero / collapsed source / unconstrained latent — PyAutoGalaxy abd7b717,
PyAutoFit #1280, PyAutoGalaxy #433) AND the interferometer-specific
failure mode where the NUFFT-via-JAX linear inversion silently returns
zero / NaN model visibilities. Lensing assertions use a deterministic
SIE sanity tracer; visibility assertions use the script's actual fit.
"""
import time as _sanity_time
from autogalaxy.operate.lens_calc import LensCalc as _SanityLensCalc

# Lensing-side: SIE sanity tracer (independent of the script's MGE model).
_sanity_lens = al.Galaxy(
    redshift=0.5,
    mass=al.mp.Isothermal(centre=(0.0, 0.0), einstein_radius=1.2, ell_comps=(0.1, 0.0)),
)
_sanity_source = al.Galaxy(redshift=1.0)
_sanity_tracer = al.Tracer(galaxies=[_sanity_lens, _sanity_source])
_sanity_od = _SanityLensCalc.from_tracer(_sanity_tracer)

_tc_list = _sanity_od.tangential_critical_curve_list_via_zero_contour_from()
assert len(_tc_list) > 0, (
    "no tangential critical curves returned by zero_contour — expected a "
    "non-empty tangential critical curve for this lens at this grid resolution"
)
_er_sanity = _sanity_od.einstein_radius_via_zero_contour_from()
assert np.isfinite(float(_er_sanity)) and float(_er_sanity) > 0.0, (
    f"Einstein radius via zero_contour returned {_er_sanity} — should be "
    "finite and positive for the SIE sanity tracer (einstein_radius=1.2)"
)
print(
    f"  PASS Visualization Sanity (lensing): "
    f"{len(_tc_list)} tangential CC, einstein_radius={float(_er_sanity):.4f}"
)

# Perf — warm-call latency must be under 100 ms (closure cache hit).
_sanity_od.tangential_critical_curve_list_via_zero_contour_from()  # warm cache
_t0 = _sanity_time.perf_counter()
_sanity_od.tangential_critical_curve_list_via_zero_contour_from()
_warm_dt = _sanity_time.perf_counter() - _t0
assert _warm_dt < 0.1, (
    f"zero_contour warm call took {_warm_dt * 1000:.1f} ms (> 100 ms) — "
    "closure cache-busting bug from PyAutoGalaxy #433 may have regressed"
)
print(f"  PASS Visualization Sanity (perf): warm call {_warm_dt * 1000:.1f} ms")

# Interferometer-specific: `fit.model_data` (an aa.Visibilities — complex
# array of model visibilities) must be finite + non-zero. Catches NUFFT /
# linear-inversion collapse on the JAX path that would leave subplot_fit.png
# cosmetically OK but the underlying visibilities all-zero.
_mv = np.asarray(fit_2.model_data)
assert np.isfinite(
    _mv
).all(), "model_data (visibilities) have nan/inf — NUFFT/inversion collapse"
assert (
    float(np.abs(_mv).sum()) > 0.0
), "model_data (visibilities) all-zero — NUFFT/inversion collapse"
print(
    f"  PASS Visualization Sanity (interferometer): "
    f"|model_data|.sum() = {float(np.abs(_mv).sum()):.4f}"
)


"""
============================================================================
Part 2 — Live Nautilus quick-update with MGE linear light profiles
============================================================================

Model: MGE linear lens (Basis of GaussianGradient + NFWSph mass) and MGE
parametric source. Linear light profiles are used, so the
``linear_light_profile_intensity_dict`` lookup is exercised during
visualization. With the ``pytree_token`` fix on ``LightProfileLinear``,
dict lookups survive the JAX pytree round-trip and no ``KeyError`` is raised.

The live search fires quick-update visualization every
``iterations_per_quick_update`` calls; we verify fit.png lands on disk.
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

lens_mge2 = af.Model(al.Galaxy, redshift=0.5, bulge=bulge_mge2, mass=mass_mge2)

source_bulge_mge2 = al.model_util.mge_model_from(
    mask_radius=mask_radius, total_gaussians=20, centre_prior_is_uniform=False
)
source_mge2 = af.Model(al.Galaxy, redshift=1.0, bulge=source_bulge_mge2)

model_mge2 = af.Collection(galaxies=af.Collection(lens=lens_mge2, source=source_mge2))


analysis_mge2 = al.AnalysisInterferometer(
    dataset=dataset,
    positions_likelihood_list=[al.PositionsLH(threshold=0.4, positions=positions)],
    use_jax=True,
)

output_root = (
    Path("scripts") / "interferometer" / "images" / "modeling_visualization_jit"
)
if output_root.exists():
    shutil.rmtree(output_root)
output_root.mkdir(parents=True)

# Also clean the autofit search output. Without this, Nautilus resumes from
# the previous run's cached samples.csv and skips live sampling — so the
# quick-update visualizer never fires, _jitted_fit_from is never set, and
# the assertion below would fail on every rerun. Force a fresh run.
output_search_root = with_test_mode_segment(Path("output")) / output_root / "mge_linear"
if output_search_root.exists():
    shutil.rmtree(output_search_root)

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
# The lens quick-update visualizer writes fit_quick.png during each quick update.
produced_pngs = list(output_search_root.rglob("fit_quick.png"))
print(f"fit_quick.png files produced: {len(produced_pngs)}")
for p in produced_pngs:
    print(f"  {p}")
assert len(produced_pngs) > 0, (
    f"no fit_quick.png produced under {output_search_root} — "
    "quick-update visualization did not fire"
)
print(
    "\nPASS: jit-cached fit_for_visualization fires during Nautilus quick updates "
    "with MGE linear profiles, fit_quick.png written, no KeyError from "
    "linear_light_profile_intensity_dict lookup."
)
