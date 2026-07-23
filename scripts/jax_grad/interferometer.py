"""
Tests JAX gradients of the interferometer log-likelihood, in two stages
(finiteness, then autodiff-vs-central-finite-difference correctness — see
``util.py``), for the two configurations used in practice:

**Variant A — parametric light profiles** (``lp.Sersic`` standard +
``lp_linear.Sersic``): the visibility-space analogue of
``imaging_lp.py``. The source is the only light in interferometer data, so
the model has no lens light component and the evaluation point is anchored
near the simulator truth so the positive-only NNLS keeps the linear source
live.

**Variant B — ``RectangularAdaptDensity`` via the sparse linear-algebra
path**: mirrors ``jax_likelihood_functions/interferometer/rectangular_sparse.py``
exactly — ``TransformerDFT`` + ``apply_sparse_operator(use_jax=True)`` (the
sparse NUFFT response-matrix formalism; the operator is aux state built once
outside the JIT trace), ``RectangularAdaptDensity`` mesh + ``reg.Adapt()`` +
``al.AdaptImages``. **Measured verdict (2026-07-09): the imaging os_pix=1
staircase applies** — interferometer pixelization has no over-sampling, so the
mesh's rank-transform queries coincide with its knots and the likelihood is
invariant to smooth mass perturbations: every mass/shear autodiff gradient is
exactly zero (correct — FD shows only ~1e-7-scale micro-jumps from rank
re-orderings, no smooth slope). With the model having no lens light, that means
**no usable gradients at all** in this configuration. The assertions document
this staircase so a change in mesh differentiability fails loudly.

**Variant D — ``RectangularKernelAdaptDensity`` via the same sparse path**
(PyAutoArray#374): the kernel-density CDF transform has no ranks or sorts, so
the staircase is structurally absent — strict FD assertions run on every
parameter in the exact configuration where variant B has no usable gradients,
plus an eager figure-of-merit parity check against the linear mesh.

**Variant C — ``RectangularUniform`` via the same sparse path**: the working
alternative for gradient-based inference — no adaptive transform, so mass/shear
gradients are live and strictly FD-matched.

See the audit README
(`autolens_workspace_developer/jax_profiling/gradient/README.md`).
"""

"""
__Env__

Test-harness configuration (PyAutoHands docs/env_profile_redesign.md §10).
Drive jax.value_and_grad + finite-difference gradient checks; need JAX
enabled and full-resolution float64 data.

ENV: jax full_datasets
"""

import numpy as np
import jax
import jax.numpy as jnp
from os import path

import autofit as af
import autolens as al

import util

"""
__Mask + Dataset__
"""
real_space_mask = al.Mask2D.circular(
    shape_native=(256, 256),
    pixel_scales=0.1,
    radius=3.0,
)

dataset_name = "simple"
dataset_path = path.join("dataset", "interferometer", dataset_name)

"""
__Dataset Auto-Simulation__
"""
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

from autofit.non_linear.fitness import Fitness


def mass_and_shear():
    """
    Truth-centred lens mass model (simulator: Isothermal ER=1.6, shear 0.05).
    """
    mass = af.Model(al.mp.Isothermal)
    mass.centre.centre_0 = af.GaussianPrior(mean=0.0, sigma=0.005)
    mass.centre.centre_1 = af.GaussianPrior(mean=0.0, sigma=0.005)
    mass.einstein_radius = af.GaussianPrior(mean=1.6, sigma=0.05)
    mass.ell_comps.ell_comps_0 = af.GaussianPrior(mean=0.05, sigma=0.01)
    mass.ell_comps.ell_comps_1 = af.GaussianPrior(mean=0.05, sigma=0.01)
    shear = af.Model(al.mp.ExternalShear)
    shear.gamma_1 = af.GaussianPrior(mean=0.05, sigma=0.005)
    shear.gamma_2 = af.GaussianPrior(mean=0.05, sigma=0.005)
    return mass, shear


def param_vector_from(model):
    param_vector = jnp.array(model.physical_values_from_prior_medians)
    key = jax.random.PRNGKey(42)
    perturbation = jax.random.uniform(
        key, shape=param_vector.shape, minval=0.001, maxval=0.005
    )
    return param_vector + perturbation


def finiteness_checks(fitness, param_vector, n_params):
    value, grad = jax.value_and_grad(fitness.call)(param_vector)
    print(f"Log likelihood = {float(value):.6f}")
    assert np.isfinite(float(value)), "Log likelihood is not finite"
    assert grad.shape == (n_params,), f"Gradient shape mismatch: {grad.shape}"
    assert np.all(
        np.isfinite(np.array(grad))
    ), f"Gradient contains non-finite values: {np.array(grad)}"
    assert not np.all(np.array(grad) == 0.0), "Gradient is all zeros"
    return grad


"""
__Variant A: parametric light profiles (standard + linear Sersic source)__
"""
for variant, bulge_cls in [
    ("lp.Sersic (standard)", al.lp.Sersic),
    ("lp_linear.Sersic (linear)", al.lp_linear.Sersic),
]:
    print(f"\n=== interferometer {variant} ===")

    source_bulge = af.Model(bulge_cls)
    source_bulge.centre.centre_0 = af.GaussianPrior(mean=0.0, sigma=0.005)
    source_bulge.centre.centre_1 = af.GaussianPrior(mean=0.0, sigma=0.005)
    source_bulge.ell_comps.ell_comps_0 = af.GaussianPrior(mean=0.05, sigma=0.01)
    source_bulge.ell_comps.ell_comps_1 = af.GaussianPrior(mean=0.05, sigma=0.01)
    source_bulge.effective_radius = af.GaussianPrior(mean=1.0, sigma=0.05)
    source_bulge.sersic_index = af.GaussianPrior(mean=1.0, sigma=0.2)

    mass, shear = mass_and_shear()
    lens = af.Model(al.Galaxy, redshift=0.5, mass=mass, shear=shear)
    source = af.Model(al.Galaxy, redshift=1.0, bulge=source_bulge)
    model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

    analysis = al.AnalysisInterferometer(dataset=dataset)

    fitness = Fitness(
        model=model,
        analysis=analysis,
        fom_is_log_likelihood=True,
        resample_figure_of_merit=-1.0e99,
    )

    param_vector = param_vector_from(model)
    param_names = util.parameter_names_from(model)

    finiteness_checks(fitness, param_vector, n_params=len(param_names))

    f_jit = jax.jit(fitness.call)

    util.assert_eager_jit_consistent(fitness.call, f_jit, param_vector)

    comparison = util.compare_gradients(
        fitness.call,
        param_vector,
        param_names=param_names,
        f_fd=f_jit,
    )

    util.assert_gradients_match(comparison)

    print(f"interferometer {variant}: autodiff matches finite differences.")

"""
__Variants B + C: rectangular pixelized source via the sparse-operator path__

The sparse NUFFT operator is aux state and must be built once, outside any JIT
trace (mirrors ``rectangular_sparse.py``).
"""
dataset_sparse = dataset.apply_sparse_operator(use_jax=True, show_progress=True)

mesh_shape = (8, 8)


def sparse_fitness(mesh, regularization):
    mass, shear = mass_and_shear()
    lens = af.Model(al.Galaxy, redshift=0.5, mass=mass, shear=shear)

    pixelization = al.Pixelization(mesh=mesh, regularization=regularization)
    source = af.Model(al.Galaxy, redshift=1.0, pixelization=pixelization)
    model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

    bulge = al.lp.Sersic()
    adapt_image = bulge.image_2d_from(grid=dataset_sparse.grid)
    adapt_images = al.AdaptImages(
        galaxy_name_image_dict={
            "('galaxies', 'lens')": adapt_image,
            "('galaxies', 'source')": adapt_image,
        }
    )

    analysis = al.AnalysisInterferometer(
        dataset=dataset_sparse,
        adapt_images=adapt_images,
        raise_inversion_positions_likelihood_exception=False,
    )

    fitness = Fitness(
        model=model,
        analysis=analysis,
        fom_is_log_likelihood=True,
        resample_figure_of_merit=-1.0e99,
    )
    return fitness, param_vector_from(model), util.parameter_names_from(model)


"""
__Variant B: RectangularAdaptDensity — the documented staircase__
"""
print("\n=== interferometer RectangularAdaptDensity + reg.Adapt, sparse operator ===")

fitness, param_vector, param_names = sparse_fitness(
    mesh=al.mesh.RectangularAdaptDensity(shape=mesh_shape),
    regularization=al.reg.Adapt(),
)

value, grad = jax.value_and_grad(fitness.call)(param_vector)
print(f"Log likelihood = {float(value):.6f}")
assert np.isfinite(float(value)), "Log likelihood is not finite"
assert np.all(
    np.isfinite(np.array(grad))
), f"Gradient contains non-finite values: {np.array(grad)}"

f_jit = jax.jit(fitness.call)

util.assert_eager_jit_consistent(fitness.call, f_jit, param_vector)

# The staircase: with no over-sampling the adaptive mesh's rank transform makes
# the likelihood invariant to smooth mass/shear perturbations, and *every* model
# parameter here is mass/shear (interferometer data has no lens light). The
# correct autodiff gradient is therefore ~zero across the board. If this
# assertion ever fails the mesh has become differentiable — rerun the full FD
# audit and update this script + the audit README.
assert np.all(np.abs(np.array(grad)) < 1e-6), (
    "Autodiff mass/shear gradients are no longer ~zero on the sparse "
    f"RectangularAdaptDensity path: {np.array(grad)}"
)

print(
    "interferometer sparse RectangularAdaptDensity: staircase confirmed — "
    "all autodiff gradients ~zero (correct; no smooth mass information)."
)

# Kept for the kernel variant's FoM parity check below (same model
# parametrization → same parameter vector).
value_linear_adapt_density = float(value)

"""
__Variant D: RectangularKernelAdaptDensity — differentiable on the sparse path__

The kernel-density CDF transform (PyAutoArray#374) replaces the empirical
point-rank CDF with ``F(x) = Σᵢ wᵢ·Φ((x−xᵢ)/h)`` — no ranks, no sorts, so the
staircase mechanism variant B documents is structurally absent. The sparse path
has no over-sampling to fall back on, which made the linear adaptive mesh's
gradients unusable here; the kernel mesh must carry live, strictly FD-matched
gradients on every (mass/shear) parameter in this exact configuration.
"""
print(
    "\n=== interferometer RectangularKernelAdaptDensity + reg.Adapt, sparse operator ==="
)

fitness, param_vector, param_names = sparse_fitness(
    mesh=al.mesh.RectangularKernelAdaptDensity(shape=mesh_shape),
    regularization=al.reg.Adapt(),
)

grad = finiteness_checks(fitness, param_vector, n_params=len(param_names))

f_jit = jax.jit(fitness.call)

util.assert_eager_jit_consistent(fitness.call, f_jit, param_vector)

# FD-step-sweep mode (see util.compare_gradients): individual FD evaluations
# are pseudo-randomly poisoned by measure-thin solver branch flips — probed
# 2026-07-10 here: LL exactly linear over ±2e-8 in gamma_2 except single float
# inputs (width < 1e-15) where the solve lands on a marginally different
# branch (ΔLL ~1.6e-3, identical for two orthogonal parameter directions;
# also present under reg.Constant, so not mesh- or reg-specific). FD converges
# to AD (rel err ≤ 1e-5) at every clean step probed over h ∈ [1e-9, 1e-5].
comparison = util.compare_gradients(
    fitness.call,
    param_vector,
    param_names=param_names,
    f_fd=f_jit,
    rel_steps=(1e-8, 1e-7, 1e-6),
)

util.assert_gradients_match(comparison)

# Every parameter here is mass/shear — all must be genuinely live (a staircase
# would pass the FD match trivially as 0 == 0).
assert np.all(np.abs(comparison["ad"]) > 1e-2), (
    "A mass/shear gradient is ~zero on the sparse RectangularKernelAdaptDensity "
    "path — the kernel mesh is not carrying smooth mass information: "
    f"{[(n, a) for n, a in zip(param_names, comparison['ad']) if abs(a) <= 1e-2]}"
)

# FoM parity vs the linear AdaptDensity mesh (variant B, same base point): the
# mesh geometry changes slightly but reconstruction quality must not degrade.
fom_kernel = float(fitness.call(param_vector))
fom_rel = abs(fom_kernel - value_linear_adapt_density) / abs(value_linear_adapt_density)
print(
    f"FoM parity: kernel = {fom_kernel:.6f}, "
    f"linear = {value_linear_adapt_density:.6f}, rel diff = {fom_rel:.3e}"
)
assert fom_rel < 5e-4, (
    f"Kernel-mesh figure_of_merit deviates from the linear mesh by {fom_rel:.3e} "
    "relative (limit 5e-4) on the sparse path — reconstruction quality has "
    "degraded; tune the mesh bandwidth."
)

print(
    "interferometer sparse RectangularKernelAdaptDensity: all gradients live, "
    "strictly FD-matched, FoM parity held."
)

"""
__Variant C: RectangularUniform — the gradient-capable alternative__
"""
print("\n=== interferometer RectangularUniform + reg.Constant, sparse operator ===")

fitness, param_vector, param_names = sparse_fitness(
    mesh=al.mesh.RectangularUniform(shape=mesh_shape),
    regularization=al.reg.Constant(coefficient=1.0),
)

grad = finiteness_checks(fitness, param_vector, n_params=len(param_names))

f_jit = jax.jit(fitness.call)

util.assert_eager_jit_consistent(fitness.call, f_jit, param_vector)

comparison = util.compare_gradients(
    fitness.call,
    param_vector,
    param_names=param_names,
    f_fd=f_jit,
)

util.assert_gradients_match(comparison)

assert np.all(np.abs(comparison["ad"]) > 0.0), (
    "A parameter gradient is exactly zero on the sparse RectangularUniform "
    f"path: {[(n, a) for n, a in zip(param_names, comparison['ad']) if a == 0.0]}"
)

print("interferometer sparse RectangularUniform: all gradients live and " "FD-matched.")

print("\ninterferometer.py JAX gradient checks passed.")
