"""
Tests JAX gradients of the interferometer log-likelihood, in two stages
(finiteness, then autodiff-vs-central-finite-difference correctness — see
``util.py``), for the configurations used in practice:

**Variant A — parametric light profiles** (``lp.Sersic`` standard +
``lp_linear.Sersic``): the visibility-space analogue of
``imaging_lp.py``. The source is the only light in interferometer data, so
the model has no lens light component and the evaluation point is anchored
near the simulator truth so the positive-only NNLS keeps the linear source
live.

**Variant B — ``RectangularRTUAdaptDensity`` via the sparse linear-algebra
path**: mirrors ``interferometer/rectangular_sparse.py``
exactly — ``TransformerDFT`` + ``apply_sparse_operator(use_jax=True)`` (the
sparse NUFFT response-matrix formalism; the operator is aux state built once
outside the JIT trace), ``RectangularRTUAdaptDensity`` mesh + ``reg.Adapt()`` +
``al.AdaptImages``. Since the rectangular-mesh consolidation
(PyAutoArray#403) this mesh IS the kernel-density-CDF mesh (formerly
``RectangularKernelAdaptDensity``, PyAutoArray#374): no ranks or sorts, so
the likelihood carries live, strictly FD-matched gradients on every
(mass/shear) parameter — on the exact path where the deleted empirical-rank
CDF was piecewise-constant with no usable gradients at all (the
interferometer pixelization has no over-sampling to fall back on; the old
staircase measurements live in the audit README).

**Variant C — ``RectangularUniform`` via the same sparse path**: no adaptive
transform, so mass/shear gradients are live and strictly FD-matched.

See the audit README
(`autolens_workspace_developer/jax_profiling/gradient/README.md`).

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

import os
import sys

# ``util.py`` lives in ``scripts/misc/`` after the mirror restructure; add it to
# the path so this script (run as a file, cwd = workspace root) can import it.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "misc"))

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
__Variant B: RectangularRTUAdaptDensity — differentiable on the sparse path__

The kernel-density CDF transform replaces the deleted empirical point-rank
CDF with ``F(x) = Σᵢ wᵢ·Φ((x−xᵢ)/h)`` — no ranks, no sorts, so the staircase
the audit documented on this path is structurally absent. The sparse path has
no over-sampling to fall back on, which made the old linear adaptive mesh's
gradients unusable here; the mesh must carry live, strictly FD-matched
gradients on every (mass/shear) parameter in this exact configuration.
"""
print("\n=== interferometer RectangularRTUAdaptDensity + reg.Adapt, sparse operator ===")

fitness, param_vector, param_names = sparse_fitness(
    mesh=al.mesh.RectangularRTUAdaptDensity(shape=mesh_shape),
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

# Every parameter here is mass/shear — all must be genuinely live (a flat
# likelihood would pass the FD match trivially as 0 == 0).
assert np.all(np.abs(comparison["ad"]) > 1e-2), (
    "A mass/shear gradient is ~zero on the sparse RectangularRTUAdaptDensity "
    "path — the kernel-CDF mesh is not carrying smooth mass information: "
    f"{[(n, a) for n, a in zip(param_names, comparison['ad']) if abs(a) <= 1e-2]}"
)

print(
    "interferometer sparse RectangularRTUAdaptDensity: all gradients live and "
    "strictly FD-matched."
)

"""
__Variant C: RectangularUniform — the non-adaptive alternative__
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
