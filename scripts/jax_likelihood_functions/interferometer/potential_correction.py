"""
JAX Likelihood: Interferometer Potential Correction (Gravitational Imaging)
===========================================================================

This script tests that the visibility-space potential-correction evidence of `al.pc` runs correctly under JAX
through the ecosystem `xp` API: the joint source+dpsi evidence of `FitDpsiSrcInterferometer` is re-derived through
the `al.pc.dense_util` kernels under `xp=numpy` and `xp=jax.numpy` (including the jitted fixed-curvature fast
path), asserting numerical agreement with both the sparse (w-tilde) and dense fit routes, and the LM kernels of
the iterative engine are checked for numpy/JAX agreement on the visibility-scale matrices.

The technique is ported from the `potential_correction` package of Cao et al. 2025
(https://github.com/caoxiaoyue/lensing_potential_correction; cite via
https://github.com/caoxiaoyue/potential_correction_paper).
"""

import numpy as np
import jax
import jax.numpy as jnp

import autoarray as aa
import autolens as al
from autolens.potential_correction import dense_util

"""
__Simulate__

A small self-contained interferometer dataset (seeded random uv coverage), Isothermal + NFW subhalo lens,
Gaussian source.
"""
rng = np.random.default_rng(1)
uv_wavelengths = rng.uniform(-3.0e5, 3.0e5, size=(600, 2))

real_space_mask = al.Mask2D.circular(
    shape_native=(48, 48), pixel_scales=0.1, radius=2.0
)

simulator = al.SimulatorInterferometer(
    uv_wavelengths=uv_wavelengths,
    exposure_time=300.0,
    noise_sigma=0.02,
    noise_seed=1,
)

lens_true = al.Galaxy(
    redshift=0.2,
    mass=al.mp.IsothermalSph(centre=(0.0, 0.0), einstein_radius=1.2),
    subhalo=al.mp.NFWMCRLudlowSph(
        centre=(1.21, 0.0), mass_at_200=1.0e10, redshift_object=0.2, redshift_source=0.6
    ),
)
source_true = al.Galaxy(
    redshift=0.6, bulge=al.lp.Gaussian(centre=(0.0, 0.0), intensity=5.0, sigma=0.2)
)

grid = al.Grid2D.from_mask(mask=real_space_mask)
dataset = simulator.via_tracer_from(
    tracer=al.Tracer(galaxies=[lens_true, source_true]), grid=grid
)
dataset = al.Interferometer(
    data=dataset.data,
    noise_map=dataset.noise_map,
    uv_wavelengths=uv_wavelengths,
    real_space_mask=real_space_mask,
)
dataset_sparse = dataset.apply_sparse_operator()

"""
__Joint fits (sparse + dense routes)__
"""
lens_smooth = al.Galaxy(redshift=0.2, mass=lens_true.mass)
source_start = al.pc.AnalyticSrcFactory(source_galaxy=source_true)

fit_kwargs = dict(
    lens_start=lens_smooth,
    source_start=source_start,
    dpsi_pixelization=al.pc.DpsiPixelization(
        mesh=al.pc.RegularDpsiMesh(factor=2),
        regularization=al.reg.MaternKernel(coefficient=100.0, scale=1.0, nu=2.5),
    ),
    src_pixelization=al.Pixelization(
        mesh=al.mesh.RectangularUniform(shape=(16, 16)),
        regularization=al.reg.Constant(coefficient=1.0),
    ),
)

fit_sparse = al.pc.FitDpsiSrcInterferometer(
    dataset=dataset_sparse, use_sparse_operator=True, **fit_kwargs
)
fit_dense = al.pc.FitDpsiSrcInterferometer(
    dataset=dataset, use_sparse_operator=False, **fit_kwargs
)

evidence_sparse = fit_sparse.log_evidence
evidence_dense = fit_dense.log_evidence
print(f"sparse-route log evidence = {evidence_sparse:.8e}")
print(f"dense-route  log evidence = {evidence_dense:.8e}")
assert np.isclose(evidence_sparse, evidence_dense, rtol=1e-3)

"""
__Dense evidence through the xp kernels: numpy vs JAX__

The dense route's stacked real/imag system is re-derived through `dense_util.log_evidence_joint_dense_from`
under both backends; the noise-normalization difference between the complex formulation and the stacked
formulation is identical, so the evidences must agree exactly.
"""
stacked_data = fit_dense._stacked_data
stacked_noise = fit_dense._stacked_noise
operated = fit_dense.operated_mapping_matrix
src_reg = np.asarray(fit_dense.src_regularization_matrix)
dpsi_reg = np.asarray(fit_dense.dpsi_regularization_matrix)

res_np = dense_util.log_evidence_joint_dense_from(
    stacked_data, stacked_noise, operated, src_reg, dpsi_reg, xp=np
)
res_jax = dense_util.log_evidence_joint_dense_from(
    stacked_data, stacked_noise, operated, src_reg, dpsi_reg, xp=jnp
)
print(f"xp=np  dense-kernel evidence = {float(res_np['evidence']):.8e}")
print(f"xp=jnp dense-kernel evidence = {float(res_jax['evidence']):.8e}")
assert bool(res_np["valid"]) and bool(res_jax["valid"])
assert np.isclose(float(res_np["evidence"]), float(res_jax["evidence"]), rtol=1e-8)
assert np.isclose(float(res_np["evidence"]), evidence_dense, rtol=1e-8)

"""
__Jitted fixed-curvature fast path__
"""
inv_var = 1.0 / stacked_noise**2
curvature = operated.T @ (operated * inv_var[:, None])
data_vector = operated.T @ (inv_var * stacked_data)
noise_term = -0.5 * float(np.sum(np.log(2 * np.pi * stacked_noise**2)))


def fixed_curvature_evidence(src_reg_in, dpsi_reg_in):
    return dense_util.log_evidence_from_fixed_curvature(
        curvature_matrix=curvature,
        data_vector=data_vector,
        data_slim=stacked_data,
        mapping_matrix=operated,
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
__LM kernels on visibility-scale matrices: numpy vs JAX__

The Marquardt-scaled LM step must agree between backends at the ~1e11 curvature magnitudes of visibility
weighting (the regime that motivated the scaling).
"""
n_s = src_reg.shape[0]
x = np.random.default_rng(0).normal(scale=0.1, size=operated.shape[1])
L = operated[:, :n_s]
J_dpsi = operated[:, n_s:]

H_np, g_np, *_ = dense_util.lm_hessian_and_gradient_from(
    stacked_data, inv_var, x, L, J_dpsi, src_reg, dpsi_reg, xp=np
)
H_jx, g_jx, *_ = dense_util.lm_hessian_and_gradient_from(
    stacked_data, inv_var, x, L, J_dpsi, src_reg, dpsi_reg, xp=jnp
)
assert np.allclose(H_np, np.asarray(H_jx), rtol=1e-10)
assert np.allclose(g_np, np.asarray(g_jx), rtol=1e-10)

step_np = dense_util.solve_lm_step_from(H_np, g_np, 1.0, xp=np)
step_jx = dense_util.solve_lm_step_from(H_jx, g_jx, 1.0, xp=jnp)
assert np.allclose(step_np, np.asarray(step_jx), rtol=1e-6)

print("interferometer potential_correction JAX likelihood checks all passed")
