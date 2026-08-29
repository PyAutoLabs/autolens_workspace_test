"""
End-to-end test: jit-cached visualization during a real Nautilus model-fit
with a DELAUNAY-pixelization source.
==========================================================================

Sibling to ``modeling_visualization_jit.py`` (MGE) and
``modeling_visualization_rectangular_jit.py`` (rectangular mesh). Exercises
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

__Env__

Test-harness configuration (PyAutoHands docs/env_profile_redesign.md §10).
JIT-cached visualization path: real search (Nautilus), JAX, full-resolution
mask and real savefig. Matches the ``modeling_visualization_jit.py`` sibling.

Without this, the script inherits the profile default ``PYAUTO_SMALL_DATASETS=1``
and regenerates the shared ``dataset/imaging/jax_test`` at 16x16 (``should_simulate``
rmtree's first when the cap is on). Every ``jax_likelihood`` script then loads that
capped data and fails its hardcoded likelihood literal, because ``should_simulate``
only tests directory existence and cannot tell the dataset was built at the wrong
size.

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

Re-use the jax_test dataset. Auto-simulate if missing.
"""
sub_size = 4

dataset_path = path.join("dataset", "imaging", "jax_test")

if al.util.dataset.should_simulate(dataset_path):
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "scripts/imaging/simulator/simple.py"],
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
    sub_size_list=[4, 2, 2],
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
``imaging/delaunay.py``.
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
)

instance_probe = model.instance_from_prior_medians()


"""
__Likelihood Sanity__

Guard against regressions like PyAutoLens PR #504, where the CPU branch of
``AnalysisImaging.log_likelihood_function`` silently returned ``fit.log_likelihood``
instead of ``fit.figure_of_merit``. For a pixelization source these differ by
the regularization log-det terms of the Bayesian log evidence, so a nested
sampler would drift to ``outer_coefficient ~= 0`` instead of the physical
Bayesian maximum.

Both backends (CPU + JAX) are checked: the bug only fired on the CPU branch
historically, but the guard catches future drift in either direction.
"""
import pytest
from autofit.non_linear.fitness import Fitness


def _assert_likelihood_sanity(label, analysis, model):
    instance = model.instance_from_prior_medians()
    analysis_value = analysis.log_likelihood_function(instance=instance)
    fit = analysis.fit_from(instance=instance)
    assert float(analysis_value) == pytest.approx(float(fit.figure_of_merit)), (
        f"{label}: log_likelihood_function ({analysis_value}) does not match "
        f"fit.figure_of_merit ({fit.figure_of_merit}) — regression of PR #504"
    )
    # ABSOLUTE tolerance, deliberately not a relative one. The Bayesian-evidence
    # correction (`figure_of_merit - log_likelihood`, i.e. the regularization and
    # log-det terms) is an absolute quantity of order tens-to-hundreds of nats set
    # by the number of source pixels; `log_likelihood` at the prior medians is set
    # by the number of DATA pixels and grows with the dataset — ~1e2 under
    # `profile_smoke.yaml`'s capped mask, ~5e8 under `profile_release.yaml`'s
    # full-resolution one. A relative tolerance therefore measures the dataset
    # scale rather than the terms. It fired for exactly that reason in PyAutoHeart
    # release-integrate run 33220882817: a perfectly healthy 391.57-nat correction
    # on log_likelihood = -5.39477770e8 is 7.26e-7 of it, inside `rel=1e-6`, so the
    # guard reported "log-det terms are zero" about a fit whose terms were nothing
    # of the kind. (The same script passed the same check in run 33177898708 only
    # because it sat on the other side of that knife edge.) `abs=1.0` states the
    # requirement the message actually claims — the terms must move the objective
    # by at least a nat — and is scale-free.
    assert float(fit.figure_of_merit) != pytest.approx(
        float(fit.log_likelihood), abs=1.0
    ), (
        f"{label}: figure_of_merit ({fit.figure_of_merit}) is within 1 nat of "
        f"log_likelihood ({fit.log_likelihood}) — pixelization regularization "
        f"log-det terms are zero, this script no longer exercises the bug PR #504 fixed"
    )
    fitness = Fitness(
        model=model,
        analysis=analysis,
        paths=None,
        fom_is_log_likelihood=True,
        resample_figure_of_merit=-1.0e99,
    )
    call_wrap_value = fitness.call_wrap(model.physical_values_from_prior_medians)
    assert float(call_wrap_value) == pytest.approx(float(fit.figure_of_merit)), (
        f"{label}: Fitness.call_wrap ({call_wrap_value}) does not match "
        f"fit.figure_of_merit ({fit.figure_of_merit})"
    )
    print(f"  PASS {label}: LLF == figure_of_merit != log_likelihood == call_wrap")


sanity_analysis_cpu = al.AnalysisImaging(
    dataset=dataset,
    adapt_images=adapt_images,
    raise_inversion_positions_likelihood_exception=False,
    use_jax=False,
)
_assert_likelihood_sanity("CPU", sanity_analysis_cpu, model)
_assert_likelihood_sanity("JAX", analysis_probe, model)


t0 = time.perf_counter()
fit_1 = analysis_probe.fit_for_visualization(instance_probe)
jax.block_until_ready(fit_1.log_likelihood)
t1 = time.perf_counter()
compile_time = t1 - t0
print(f"First call (compile + run): {compile_time:.3f}s")
print(f"  log_likelihood leaf type: {type(fit_1.log_likelihood).__name__}")
assert isinstance(
    fit_1.log_likelihood, jnp.ndarray
), f"expected jax.Array, got {type(fit_1.log_likelihood)}"

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
print("PASS: Delaunay jit-cached fit_for_visualization works and is reused.")


"""
__Visualization Sanity__

Phase D.1 rollout of the Sanity-block pattern from PR #111. Guards
against the silent-zero / collapsed-source / unconstrained-latent
regression class on the JIT-cached visualization path, for the
pixelization-source variant where the script's prior-median model is
even less likely to produce strong-enough lensing than the parametric
imaging case. Uses a deterministic SIE sanity tracer so the assertions
are independent of the script's pixelization configuration.
"""
import time as _sanity_time
from autogalaxy.operate.lens_calc import LensCalc as _SanityLensCalc

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
    f"  PASS Visualization Sanity (correctness): "
    f"{len(_tc_list)} tangential CC, einstein_radius={float(_er_sanity):.4f}"
)

_sanity_od.tangential_critical_curve_list_via_zero_contour_from()  # warm cache
_t0 = _sanity_time.perf_counter()
_sanity_od.tangential_critical_curve_list_via_zero_contour_from()
_warm_dt = _sanity_time.perf_counter() - _t0
assert _warm_dt < 0.1, (
    f"zero_contour warm call took {_warm_dt * 1000:.1f} ms (> 100 ms) — "
    "closure cache-busting bug from PyAutoGalaxy #433 may have regressed"
)
print(f"  PASS Visualization Sanity (perf): warm call {_warm_dt * 1000:.1f} ms")


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
)

output_root = (
    Path("scripts") / "imaging" / "images" / "modeling_visualization_delaunay_jit"
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

output_search_root = with_test_mode_segment(Path("output")) / output_root / "delaunay"
produced_pngs = list(output_search_root.rglob("fit.png"))
print(f"fit.png files produced: {len(produced_pngs)}")
for p in produced_pngs:
    print(f"  {p}")
assert len(produced_pngs) > 0, (
    f"no fit.png produced under {output_search_root} — "
    "quick-update visualization did not fire"
)
print(
    "\nPASS: jit-cached fit_for_visualization fires during Nautilus quick updates "
    "with a Delaunay-pixelization source, fit.png written."
)
