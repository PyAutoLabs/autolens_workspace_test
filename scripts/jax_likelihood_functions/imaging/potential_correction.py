"""
JAX Likelihood: Potential Correction (Gravitational Imaging)
============================================================

This script tests that the potential-correction (gravitational imaging) evidence of `al.pc` runs correctly under
JAX through the ecosystem `xp` API: every dense kernel of `al.pc.dense_util` is written once with an `xp=np`
default, and this script evaluates the joint source+dpsi evidence, the fixed-curvature fast path and the
Levenberg-Marquardt kernels under `xp=jax.numpy`, asserting numerical agreement with the numpy path and
jit-compilability.

The technique is ported from the `potential_correction` package of Cao et al. 2025
(https://github.com/caoxiaoyue/lensing_potential_correction; cite via
https://github.com/caoxiaoyue/potential_correction_paper).

__Env__

Test-harness configuration (PyAutoHands docs/env_profile_redesign.md §10).
JAX likelihood functions test JIT compilation; need JAX enabled and full-
size datasets.

ENV: jax full_datasets
"""

import numpy as np
import jax
import jax.numpy as jnp

import autoarray as aa
import autolens as al
from autolens.potential_correction import dense_util

"""
__Simulate__

A small self-contained lens: Isothermal + NFW subhalo on the Einstein ring, double-Gaussian source. The dataset is
simulated in-memory (seeded), keeping this script dependency-free.
"""
grid = al.Grid2D.uniform(shape_native=(60, 60), pixel_scales=0.1, over_sample_size=4)
psf = al.Convolver.from_gaussian(shape_native=(7, 7), sigma=0.1, pixel_scales=0.1)

simulator = al.SimulatorImaging(
    exposure_time=840.0,
    psf=psf,
    background_sky_level=0.1,
    add_poisson_noise_to_data=True,
    noise_seed=1,
)

lens_true = al.Galaxy(
    redshift=0.2,
    mass=al.mp.IsothermalSph(centre=(0.0, 0.0), einstein_radius=1.4),
    subhalo=al.mp.NFWMCRLudlowSph(
        centre=(1.41, 0.0), mass_at_200=1.0e10, redshift_object=0.2, redshift_source=0.6
    ),
)
source_true = al.Galaxy(
    redshift=0.6,
    bulge=al.lp.Gaussian(centre=(0.0, 0.0), intensity=5.0, sigma=0.2),
)
dataset = simulator.via_tracer_from(
    tracer=al.Tracer(galaxies=[lens_true, source_true]), grid=grid
)

mask_array = al.pc.util.arc_mask_from(
    np.asarray(dataset.signal_to_noise_map.native),
    threshold=3.0,
    ignore_size=10,
    ext_size=3,
)
masked_imaging = dataset.apply_mask(
    mask=al.Mask2D(mask=mask_array, pixel_scales=dataset.pixel_scales)
)

"""
__Joint Fit (numpy reference)__

The one-shot joint source+dpsi fit at the smooth (subhalo-free) starting model provides the reference matrices.
"""
lens_smooth = al.Galaxy(redshift=0.2, mass=lens_true.mass)
source_start = al.pc.AnalyticSrcFactory(source_galaxy=source_true)

fit = al.pc.FitDpsiSrcImaging(
    masked_imaging=masked_imaging,
    lens_start=lens_smooth,
    source_start=source_start,
    dpsi_pixelization=al.pc.DpsiPixelization(
        mesh=al.pc.RegularDpsiMesh(factor=2),
        regularization=al.reg.MaternKernel(coefficient=100.0, scale=1.0, nu=2.5),
    ),
    src_pixelization=al.Pixelization(
        mesh=al.mesh.RectangularUniform(shape=(20, 20)),
        regularization=al.reg.Constant(coefficient=1.0),
    ),
)

evidence_np_fit = fit.log_evidence
print(f"numpy sparse-path log evidence = {evidence_np_fit:.8e}")

data = np.asarray(masked_imaging.data)
noise = np.asarray(masked_imaging.noise_map)
mapping = np.asarray(fit.mapping_matrix)
src_reg = np.asarray(fit.src_regularization_matrix)
dpsi_reg = np.asarray(fit.dpsi_regularization_matrix)

"""
__Dense Evidence: numpy vs JAX__

The same evidence through the dense `xp` kernels, under both backends. All three numbers must agree.
"""
res_np = dense_util.log_evidence_joint_dense_from(
    data, noise, mapping, src_reg, dpsi_reg, xp=np
)
res_jax = dense_util.log_evidence_joint_dense_from(
    data, noise, mapping, src_reg, dpsi_reg, xp=jnp
)

print(f"dense evidence xp=np  = {float(res_np['evidence']):.8e}")
print(f"dense evidence xp=jnp = {float(res_jax['evidence']):.8e}")

assert bool(res_np["valid"]) and bool(res_jax["valid"])
assert np.isclose(float(res_np["evidence"]), evidence_np_fit, rtol=1e-8)
assert np.isclose(float(res_np["evidence"]), float(res_jax["evidence"]), rtol=1e-10)

"""
__Fixed-Curvature Fast Path + JIT__

The fixed-curvature kernel (curvature matrix precomputed; only the regularizations change per sample) is the hot
path of evidence-based hyper-parameter sampling. It must agree with the full evidence and be jit-compilable with
the regularization matrices as traced inputs.
"""
inv_var = 1.0 / noise**2
curvature = mapping.T @ (mapping * inv_var[:, None])
data_vector = mapping.T @ (inv_var * data)
noise_term = -0.5 * float(np.sum(np.log(2 * np.pi * noise**2)))


def fixed_curvature_evidence(src_reg_in, dpsi_reg_in):
    return dense_util.log_evidence_from_fixed_curvature(
        curvature_matrix=curvature,
        data_vector=data_vector,
        data_slim=data,
        mapping_matrix=mapping,
        inv_var=inv_var,
        noise_term=noise_term,
        src_reg_matrix=src_reg_in,
        dpsi_reg_matrix=dpsi_reg_in,
        xp=jnp,
    )["evidence"]


evidence_jitted = jax.jit(fixed_curvature_evidence)(src_reg, dpsi_reg)
print(f"jitted fixed-curvature evidence = {float(evidence_jitted):.8e}")
assert np.isclose(float(evidence_jitted), float(res_np["evidence"]), rtol=1e-8)

"""
__LM Kernels under JAX__

The Levenberg-Marquardt kernels of the iterative engine, evaluated under both backends at a random state.
"""
rng = np.random.default_rng(0)
n_s = src_reg.shape[0]
x = rng.normal(scale=0.1, size=mapping.shape[1])
L = mapping[:, :n_s]
J_dpsi = mapping[:, n_s:]

H_np, g_np, *_ = dense_util.lm_hessian_and_gradient_from(
    data, inv_var, x, L, J_dpsi, src_reg, dpsi_reg, xp=np
)
H_jx, g_jx, *_ = dense_util.lm_hessian_and_gradient_from(
    data, inv_var, x, L, J_dpsi, src_reg, dpsi_reg, xp=jnp
)
assert np.allclose(H_np, np.asarray(H_jx), rtol=1e-10)
assert np.allclose(g_np, np.asarray(g_jx), rtol=1e-10)

step_np = dense_util.solve_lm_step_from(H_np, g_np, 1.0, xp=np)
step_jx = dense_util.solve_lm_step_from(H_jx, g_jx, 1.0, xp=jnp)
assert np.allclose(step_np, np.asarray(step_jx), rtol=1e-8)

print("potential_correction JAX likelihood checks all passed")
