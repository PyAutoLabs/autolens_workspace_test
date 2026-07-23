"""
Visualization JAX Pilot: Imaging Analysis
=========================================

Pilot for https://github.com/PyAutoLabs/PyAutoFit/issues/1227.

Goal
----
Run ``VisualizerImaging.visualize`` with JAX enabled end-to-end via
``use_jax=True`` on ``Analysis``. After PyAutoLens #443 (2026-04-19) the
imaging visualizer dispatches through ``analysis.fit_for_visualization``,
which lazily wraps ``fit_from`` in ``jax.jit``. Visualization now follows
``use_jax`` automatically. To trace across that boundary the model and fit
return type must be JAX pytrees, so this script enables pytree registration
before constructing the model. Parametric MGE source — simplest case (no
pixelization, no inversion).

Scope
-----
- Parametric MGE source only.
- Calls ``VisualizerImaging.visualize`` only (not ``visualize_before_fit``).
- Re-uses the ``jax_test`` dataset from ``jax_likelihood_functions/imaging``.
- Reuses ``config_source/visualize/plots.yaml`` from ``visualization.py`` so
  only ``fit.png`` and ``tracer.png`` are attempted.
"""
# ENV: jax full_datasets real_plots
# JIT-cached fit_for_visualization path; needs JAX enabled, real
# plots and full-resolution data.

import shutil
from os import path
from pathlib import Path
from types import SimpleNamespace

from autolens import conf

conf.instance.push(
    new_path=path.join(path.dirname(path.realpath(__file__)), "config_source"),
    output_path=path.join(path.dirname(path.realpath(__file__)), "images"),
)

import autofit as af
import autolens as al
from autolens.imaging.model.visualizer import VisualizerImaging


"""
__Dataset__

Re-use the jax_test dataset already used by ``jax_likelihood_functions/imaging``.
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

mask_radius = 3.0
mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=mask_radius,
)
dataset = dataset.apply_mask(mask=mask)


"""
__Model__

MGE parametric lens + MGE parametric source (matches the MGE pattern in
``jax_likelihood_functions/imaging/mge.py``).
"""
lens_bulge = al.model_util.mge_model_from(
    mask_radius=mask_radius, total_gaussians=20, centre_prior_is_uniform=True
)
mass = af.Model(al.mp.PowerLaw)
mass.centre.centre_0 = 0.0
mass.centre.centre_1 = 0.0
mass.ell_comps.ell_comps_0 = 0.05
mass.ell_comps.ell_comps_1 = 0.1
mass.einstein_radius = 1.6
mass.slope = 2.0

lens = af.Model(al.Galaxy, redshift=0.5, bulge=lens_bulge, mass=mass)

source_bulge = al.model_util.mge_model_from(
    mask_radius=mask_radius, total_gaussians=20, centre_prior_is_uniform=False
)
source = af.Model(al.Galaxy, redshift=1.0, bulge=source_bulge)

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))


"""
__Analysis__

``use_jax=True`` turns on the JAX ``_xp`` path. Visualization now follows
``use_jax`` automatically via the ``Analysis.fit_for_visualization`` helper.
"""
analysis = al.AnalysisImaging(
    dataset=dataset,
    use_jax=True,
    title_prefix="JAX_PILOT",
)


"""
__Paths__
"""
image_path = Path("scripts") / "imaging" / "images" / "visualization_jax"
if image_path.exists():
    shutil.rmtree(image_path)
image_path.mkdir(parents=True)
output_path = image_path / "output"
output_path.mkdir(parents=True)
paths = SimpleNamespace(image_path=image_path, output_path=output_path)


"""
__Run visualize on the eager-JAX fit__
"""
instance = model.instance_from_prior_medians()

print("Running VisualizerImaging.visualize with use_jax=True ...")
VisualizerImaging.visualize(
    analysis=analysis,
    paths=paths,
    instance=instance,
    during_analysis=False,
)
assert (image_path / "parametric" / "fit.png").exists() or (
    image_path / "fit.png"
).exists(), "fit.png was not produced"
print("PILOT SUCCEEDED — JAX-backed visualization produced fit.png/tracer.png.")


"""
__Visualization Sanity__

Phase D.2.a rollout of the Sanity-block pattern from PR #111 / #113.
Same imaging-template SIE assertions as the modeling_visualization_jit
variant — catches the silent-zero / cache-busting failure class on the
single-shot JAX-backed visualization path. Uses a deterministic SIE
tracer so assertions are independent of the script's specific model.
"""
import time as _sanity_time
import numpy as _sanity_np
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
assert _sanity_np.isfinite(float(_er_sanity)) and float(_er_sanity) > 0.0, (
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
