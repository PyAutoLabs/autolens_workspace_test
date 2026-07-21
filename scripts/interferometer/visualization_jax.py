"""
Visualization JAX Pilot: Interferometer Analysis
=================================================

Pilot for https://github.com/PyAutoLabs/PyAutoFit/issues/1227.

Goal
----
Run ``VisualizerInterferometer.visualize`` with JAX enabled end-to-end via
``use_jax=True`` on ``Analysis``. After PyAutoLens #443 the interferometer
visualizer dispatches through ``analysis.fit_for_visualization``, which lazily
wraps ``fit_from`` in ``jax.jit``
(autolens/interferometer/model/visualizer.py:96). Visualization now follows
``use_jax`` automatically. To trace across that boundary the model and fit
return type must be JAX pytrees, so this script enables pytree registration
before constructing the model. Parametric MGE source — simplest case (no PSF
convolution; interferometer operates in Fourier space via DFT, no
pixelization, no inversion).

Scope
-----
- Parametric MGE source only.
- Calls ``VisualizerInterferometer.visualize`` only (not ``visualize_before_fit``).
- Re-uses the ``simple`` dataset from ``jax_likelihood_functions/interferometer``.
- Uses the default plot config (no bespoke config_source override).
"""

import shutil
from os import path
from pathlib import Path
from types import SimpleNamespace

import autofit as af
import autolens as al
from autolens.interferometer.model.visualizer import VisualizerInterferometer


"""
__Dataset__

Re-use the ``simple`` interferometer dataset used by
``jax_likelihood_functions/interferometer``. Auto-simulate if missing.
"""
mask_radius = 3.0

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
            "scripts/jax_likelihood_functions/interferometer/simulator.py",
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
__Model__

Lens: Isothermal mass + ExternalShear (matching the interferometer mge.py
pattern; no lens light). Source: MGE parametric bulge.
"""
mass = af.Model(al.mp.Isothermal)
shear = af.Model(al.mp.ExternalShear)
lens = af.Model(al.Galaxy, redshift=0.5, mass=mass, shear=shear)

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
analysis = al.AnalysisInterferometer(
    dataset=dataset,
    positions_likelihood_list=[al.PositionsLH(threshold=0.4, positions=positions)],
    use_jax=True,
    title_prefix="JAX_PILOT",
)


"""
__Paths__
"""
image_path = Path("scripts") / "interferometer" / "images" / "visualization_jax"
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

print("Running VisualizerInterferometer.visualize with use_jax=True ...")
VisualizerInterferometer.visualize(
    analysis=analysis,
    paths=paths,
    instance=instance,
    during_analysis=False,
)
assert (image_path / "fit.png").exists(), "fit.png was not produced"
print(
    "PILOT SUCCEEDED — JAX-backed interferometer visualization produced fit.png/tracer.png."
)


"""
__Visualization Sanity__

Phase D.2.a rollout. Lensing-side SIE Sanity (silent-zero / cache-busting
regression class) plus interferometer-specific `fit.model_data` (complex
visibilities) finite + non-zero on the script's actual fit.
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

_sanity_t0 = _sanity_time.perf_counter()
_tc_list = (
    _sanity_od.tangential_critical_curve_list_via_zero_contour_from()
)  # cold: first call on fresh instance (JIT compile)
_sanity_cold_dt = _sanity_time.perf_counter() - _sanity_t0
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
    f"  PASS Visualization Sanity (lensing): "
    f"{len(_tc_list)} tangential CC, einstein_radius={float(_er_sanity):.4f}"
)

_t0 = _sanity_time.perf_counter()
_sanity_od.tangential_critical_curve_list_via_zero_contour_from()  # warm (cached solver)
_warm_dt = _sanity_time.perf_counter() - _t0
# Hardware-independent guard: the warm (cached) call must be much faster than
# the cold JIT-compiling first call. A closure cache-busting regression
# (PyAutoGalaxy #433) recompiles the solver every call, so warm ~= cold (ratio
# near 1). An absolute millisecond budget instead false-positives on slower
# machines where the honest warm call legitimately exceeds it. Mirrors the
# compile-vs-cached ratio guard used elsewhere in this suite.
assert _warm_dt < _sanity_cold_dt * 0.5, (
    f"zero_contour warm call {_warm_dt * 1000:.1f} ms vs cold "
    f"{_sanity_cold_dt * 1000:.1f} ms (ratio {_warm_dt / _sanity_cold_dt:.2f}) — "
    "warm should be much faster than the cold compile; a ratio near 1 means the "
    "closure cache-busting bug from PyAutoGalaxy #433 has regressed"
)
print(
    f"  PASS Visualization Sanity (perf): warm {_warm_dt * 1000:.1f} ms vs "
    f"cold {_sanity_cold_dt * 1000:.1f} ms (ratio {_warm_dt / _sanity_cold_dt:.2f})"
)

# Interferometer-specific: model_data (complex Visibilities) finite + non-zero.
_fit_for_vis = analysis.fit_from(instance=instance)
_mv = _sanity_np.asarray(_fit_for_vis.model_data)
assert _sanity_np.isfinite(
    _mv
).all(), "fit.model_data (visibilities) have nan/inf — NUFFT/inversion collapse"
assert (
    float(_sanity_np.abs(_mv).sum()) > 0.0
), "fit.model_data (visibilities) all-zero — NUFFT/inversion collapse"
print(
    f"  PASS Visualization Sanity (interferometer): "
    f"|model_data|.sum() = {float(_sanity_np.abs(_mv).sum()):.4f}"
)
