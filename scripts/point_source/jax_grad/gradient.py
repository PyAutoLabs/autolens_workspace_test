"""
Tests JAX gradients of the point-source chi-squared variants — the solver-free
**source-plane** fits (``al.FitPositionsSource`` / ``...Solved``) and the
solver-chained **image-plane** fits (``al.FitPositionsImagePairAll`` /
``...Solved``, via the ``PointSolver`` implicit-diff ``custom_jvp``) — in two
stages:

 1. Finiteness — ``jax.value_and_grad`` returns a finite log-likelihood and a
    finite, non-zero gradient vector.
 2. Correctness — the autodiff gradient agrees with central finite differences
    parameter-by-parameter (see ``util.py``).

The source-plane likelihood ray-traces the observed image-plane positions to
the source plane and penalises their distance to the modelled source centre,
weighted by per-position magnifications computed from the Hessian of the
deflection field — so this validates second-derivative flow through the mass
profiles.

Notable: the forward ``jax.jit`` of this likelihood is blocked
(``Grid2DIrregular.grid_2d_via_deflection_grid_from`` does not propagate
``xp``), but ``jax.value_and_grad`` succeeds — the gradient path is usable by
samplers today even though the jitted forward pass is not. Because of that,
the finite-difference sweep here uses the eager likelihood (no jit), unlike
the imaging scripts.

Parameters with no positional information (``point_0.flux``, ``H0``) have
legitimately zero gradients — both autodiff and finite differences agree on
zero, and the script asserts the positional parameters are live.

Setup mirrors ``scripts/point_source/source_plane.py``.

__Env__

Test-harness configuration (PyAutoHands docs/env_profile_redesign.md §10).
Drive jax.value_and_grad + finite-difference gradient checks; need JAX
enabled and full-resolution float64 data.

ENV: jax full_datasets
"""

import numpy as np
import jax
import jax.numpy as jnp
from pathlib import Path

import autofit as af
import autolens as al

import os
import sys

# ``util.py`` lives in ``scripts/misc/`` after the mirror restructure; add it to
# the path so this script (run as a file, cwd = workspace root) can import it.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "misc"))

import util

"""
__Dataset__
"""
dataset_name = "simple"
dataset_path = Path("dataset") / "point_source" / dataset_name

"""
__Dataset Auto-Simulation__
"""
if al.util.dataset.should_simulate(str(dataset_path)):
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "scripts/point_source/simulators/simple.py"],
        check=True,
    )

dataset = al.from_json(
    file_path=dataset_path / "point_dataset_positions_only.json",
)

"""
__Point Solver__

Source-plane chi-squared does not use the solver, but ``AnalysisPoint``
requires one — pass the standard JAX-friendly solver for consistency.
"""
grid = al.Grid2D.uniform(
    shape_native=(100, 100),
    pixel_scales=0.2,
)

solver = al.PointSolver.for_grid(
    grid=grid, pixel_scale_precision=0.001, magnification_threshold=0.1
)

"""
__Model__
"""
mass = af.Model(al.mp.Isothermal)

mass.centre.centre_0 = af.UniformPrior(lower_limit=0.0, upper_limit=0.02)
mass.centre.centre_1 = af.UniformPrior(lower_limit=0.0, upper_limit=0.02)
mass.ell_comps.ell_comps_0 = af.UniformPrior(lower_limit=0.0, upper_limit=0.02)
mass.ell_comps.ell_comps_1 = af.UniformPrior(lower_limit=0.0, upper_limit=0.02)
mass.einstein_radius = af.UniformPrior(lower_limit=1.5, upper_limit=1.8)

lens = af.Model(al.Galaxy, redshift=0.5, mass=mass)

point_0 = af.Model(al.ps.PointFlux)
point_0.centre.centre_0 = af.UniformPrior(lower_limit=0.06, upper_limit=0.08)
point_0.centre.centre_1 = af.UniformPrior(lower_limit=0.06, upper_limit=0.08)

source = af.Model(al.Galaxy, redshift=1.0, point_0=point_0)

cosmology = af.Model(al.cosmo.FlatLambdaCDM)
cosmology.H0 = af.UniformPrior(lower_limit=0.0, upper_limit=150.0)

model = af.Collection(
    galaxies=af.Collection(lens=lens, source=source), cosmology=cosmology
)

print(model.info)

"""
__Analysis__
"""
analysis = al.AnalysisPoint(
    dataset=dataset,
    solver=solver,
    fit_positions_cls=al.FitPositionsSource,
)

from autofit.non_linear.fitness import Fitness

fitness = Fitness(
    model=model,
    analysis=analysis,
    fom_is_log_likelihood=True,
    resample_figure_of_merit=-1.0e99,
)

param_vector = jnp.array(model.physical_values_from_prior_medians)

key = jax.random.PRNGKey(42)
perturbation = jax.random.uniform(
    key, shape=param_vector.shape, minval=0.001, maxval=0.005
)
param_vector = param_vector + perturbation

value, grad = jax.value_and_grad(fitness.call)(param_vector)

print(f"Log likelihood = {float(value):.6f}")
print(f"Gradient shape = {grad.shape}")

assert np.isfinite(float(value)), "Log likelihood is not finite"
assert grad.shape == (
    model.total_free_parameters,
), f"Gradient shape mismatch: {grad.shape}"
assert np.all(
    np.isfinite(np.array(grad))
), f"Gradient contains non-finite values: {np.array(grad)}"
assert not np.all(np.array(grad) == 0.0), "Gradient is all zeros"

"""
__Finite-Difference Correctness__

Eager evaluation throughout — the forward jit of this likelihood is blocked
(see module docstring), which the FD sweep tolerates because the positions
dataset is tiny.
"""
param_names = util.parameter_names_from(model)

comparison = util.compare_gradients(
    fitness.call,
    param_vector,
    param_names=param_names,
)

util.assert_gradients_match(comparison)

# The positional parameters (everything except flux and H0) must be live.
positional_indices = [
    i for i, name in enumerate(param_names) if "flux" not in name and "H0" not in name
]
assert np.all(
    np.abs(comparison["ad"][positional_indices]) > 0.0
), "A positional parameter has zero gradient — evaluation point is degenerate."

print("point_source.py JAX gradient checks passed.")


"""
__Solved Source-Plane Gradient (Parameter-Free Centre)__

Repeats the finiteness + finite-difference checks above for
``al.FitPositionsSourceSolved`` against a parameter-free ``al.ps.PointSolved``
source. With no source-centre parameters, the model's only positional degrees
of freedom are the lens mass parameters; ``cosmology.H0`` still has no
bearing on this position-only chi-squared and stays possibly-zero-grad.

Gradients through the image-plane ``PointSolver`` variants are certified in the
blocks below via the solver's implicit-diff ``custom_jvp``
(``autolens.point.solver.implicit_diff``, phase 5 of issue #657).
"""

point_0_solved = af.Model(al.ps.PointSolved)

source_solved = af.Model(al.Galaxy, redshift=1.0, point_0=point_0_solved)

model_solved = af.Collection(
    galaxies=af.Collection(lens=lens, source=source_solved), cosmology=cosmology
)

print(model_solved.info)

analysis_solved = al.AnalysisPoint(
    dataset=dataset,
    solver=solver,
    fit_positions_cls=al.FitPositionsSourceSolved,
)

fitness_solved = Fitness(
    model=model_solved,
    analysis=analysis_solved,
    fom_is_log_likelihood=True,
    resample_figure_of_merit=-1.0e99,
)

param_vector_solved = jnp.array(model_solved.physical_values_from_prior_medians)

key_solved = jax.random.PRNGKey(43)
perturbation_solved = jax.random.uniform(
    key_solved, shape=param_vector_solved.shape, minval=0.001, maxval=0.005
)
param_vector_solved = param_vector_solved + perturbation_solved

value_solved, grad_solved = jax.value_and_grad(fitness_solved.call)(param_vector_solved)

print(f"Log likelihood (solved) = {float(value_solved):.6f}")
print(f"Gradient shape (solved) = {grad_solved.shape}")

assert np.isfinite(float(value_solved)), "Log likelihood (solved) is not finite"
assert grad_solved.shape == (
    model_solved.total_free_parameters,
), f"Gradient shape mismatch (solved): {grad_solved.shape}"
assert np.all(
    np.isfinite(np.array(grad_solved))
), f"Gradient contains non-finite values (solved): {np.array(grad_solved)}"
assert not np.all(np.array(grad_solved) == 0.0), "Gradient is all zeros (solved)"

param_names_solved = util.parameter_names_from(model_solved)

comparison_solved = util.compare_gradients(
    fitness_solved.call,
    param_vector_solved,
    param_names=param_names_solved,
)

util.assert_gradients_match(comparison_solved)

# With PointSolved there are no source-centre parameters — the only
# positional degrees of freedom are the lens mass parameters. H0 must still
# be excluded (this position-only chi-squared has no dependence on it).
positional_indices_solved = [
    i for i, name in enumerate(param_names_solved) if "H0" not in name
]
assert np.all(
    np.abs(comparison_solved["ad"][positional_indices_solved]) > 0.0
), "A positional parameter has zero gradient — evaluation point is degenerate (solved)."

print("point_source gradient.py solved-source checks passed.")


"""
__Image-Plane Solver Gradients (implicit-diff custom_jvp)__

The blocks below certify gradients THROUGH the ``PointSolver`` forward solve —
possible because the solver applies the implicit fixed-point rule
``A dtheta = dalpha + dbeta`` at its solved positions
(``autolens.point.solver.implicit_diff``; the gravity.jl / Lombardi 2024 Eq. 30
mechanism — differentiate at the solution, never through the triangle
refinement).

**FD methodology — the solver staircase.** The forward solve quantizes
positions at ``pixel_scale_precision``, so the computed likelihood is a
staircase and central differences BELOW the stair width read exactly zero.
The implicit gradient is the derivative of the exact-solve envelope — the
quantity a gradient search follows. Certification therefore uses:

 1. a fine-precision solver (``pixel_scale_precision=1e-5``) so the stairs are
    ~100x smaller than production, and
 2. a per-parameter FD step sweep (``rel_steps``, the interferometer/delaunay
    convention) whose steps span many stairs, with ``FD_SOLVER_RTOL`` (2%) —
    residual stair noise sits at the ~1-2% level at a generic base point, while
    a WRONG implicit rule would miss at every parameter and every step, not sit
    at the stair-noise floor.

**Known limitation — free cosmology.** ``Tracer`` registers ``cosmology`` as
aux (``no_flatten``), so a FREE cosmology parameter rides the ``custom_jvp``
boundary as a stale tracer and raises ``UnexpectedTracerError``. The
solver-chained models below therefore carry no cosmology component — physically
lossless here (H0 does not move 2-plane image positions). Multi-plane gradient
fits with free cosmology need the cosmology flattened into the Tracer pytree
(requires registered cosmology classes) — recorded as a phase-5 follow-up on
issue #657.
"""
solver_fine = al.PointSolver.for_grid(
    grid=grid, pixel_scale_precision=1e-5, magnification_threshold=0.1
)

FD_REL_STEPS = (1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3)

# rtol for the solver-chained FD comparisons: residual stair noise on the FD side sits
# at the ~1-2% level at a generic base point (the AD value is step-free; the FD sweep
# straddles solver quanta). A WRONG implicit rule — e.g. a dropped dbeta term or an
# untransposed Jacobian — misses by factors, at every parameter, at every step.
FD_SOLVER_RTOL = 2e-2

"""
__Block C — FitPositionsImagePairAllSolved + PointSolved (solved centre through the solver)__

The all-pairs mixture is smooth in the pairings (LogSumExp), so with solver
gradients the whole chain params -> beta* -> solver -> mixture is
differentiable. The model has no source parameters at all.
"""
model_ip_solved = af.Collection(
    galaxies=af.Collection(lens=lens, source=source_solved)
)

analysis_ip_solved = al.AnalysisPoint(
    dataset=dataset,
    solver=solver_fine,
    fit_positions_cls=al.FitPositionsImagePairAllSolved,
)

fitness_ip_solved = Fitness(
    model=model_ip_solved,
    analysis=analysis_ip_solved,
    fom_is_log_likelihood=True,
    resample_figure_of_merit=-1.0e99,
)

param_vector_ip = jnp.array(model_ip_solved.physical_values_from_prior_medians)
key_ip = jax.random.PRNGKey(44)
param_vector_ip = param_vector_ip + jax.random.uniform(
    key_ip, shape=param_vector_ip.shape, minval=0.0002, maxval=0.0006
)

value_ip, grad_ip = jax.value_and_grad(fitness_ip_solved.call)(param_vector_ip)

print(f"Log likelihood (image-plane solved) = {float(value_ip):.6f}")

assert np.isfinite(float(value_ip)), "Log likelihood (image-plane solved) is not finite"
assert np.all(
    np.isfinite(np.array(grad_ip))
), f"Gradient contains non-finite values (image-plane solved): {np.array(grad_ip)}"
assert not np.all(
    np.array(grad_ip) == 0.0
), "Gradient is all zeros (image-plane solved) — the solver custom_jvp did not engage"

param_names_ip = util.parameter_names_from(model_ip_solved)

comparison_ip = util.compare_gradients(
    fitness_ip_solved.call,
    param_vector_ip,
    param_names=param_names_ip,
    rel_steps=FD_REL_STEPS,
)

util.assert_gradients_match(comparison_ip, rtol=FD_SOLVER_RTOL)

# Every parameter is a lens mass parameter — all must be live through the solver.
assert np.all(
    np.abs(comparison_ip["ad"]) > 0.0
), "A mass parameter has zero gradient through the solver (image-plane solved)."

print("point_source gradient.py image-plane SOLVED-centre checks passed.")

"""
__Block D — FitPositionsImagePairAll + Point (sampled centre through the solver)__

The centre-sampled twin: the source centre enters the solver as beta, so this
exercises the ``dbeta`` term of the implicit rule (block C's beta* covers the
solved route). Uses the free-centre model from the first block (``PointFlux``;
its ``flux`` has legitimately zero gradient in a positions-only fit).
"""
model_ip_free = af.Collection(galaxies=af.Collection(lens=lens, source=source))

analysis_ip_free = al.AnalysisPoint(
    dataset=dataset,
    solver=solver_fine,
    fit_positions_cls=al.FitPositionsImagePairAll,
)

fitness_ip_free = Fitness(
    model=model_ip_free,
    analysis=analysis_ip_free,
    fom_is_log_likelihood=True,
    resample_figure_of_merit=-1.0e99,
)

param_vector_ipf = jnp.array(model_ip_free.physical_values_from_prior_medians)
key_ipf = jax.random.PRNGKey(45)
param_vector_ipf = param_vector_ipf + jax.random.uniform(
    key_ipf, shape=param_vector_ipf.shape, minval=0.0002, maxval=0.0006
)

value_ipf, grad_ipf = jax.value_and_grad(fitness_ip_free.call)(param_vector_ipf)

print(f"Log likelihood (image-plane free centre) = {float(value_ipf):.6f}")

assert np.isfinite(float(value_ipf)), "Log likelihood (image-plane free) is not finite"
assert np.all(
    np.isfinite(np.array(grad_ipf))
), f"Gradient contains non-finite values (image-plane free): {np.array(grad_ipf)}"

param_names_ipf = util.parameter_names_from(model_ip_free)

comparison_ipf = util.compare_gradients(
    fitness_ip_free.call,
    param_vector_ipf,
    param_names=param_names_ipf,
    rel_steps=FD_REL_STEPS,
)

util.assert_gradients_match(comparison_ipf, rtol=FD_SOLVER_RTOL)

# flux carries no positional information; centre + mass must be live.
positional_indices_ipf = [
    i for i, name in enumerate(param_names_ipf) if "flux" not in name
]
assert np.all(
    np.abs(comparison_ipf["ad"][positional_indices_ipf]) > 0.0
), "A positional parameter has zero gradient through the solver (image-plane free)."

print("point_source gradient.py image-plane FREE-centre checks passed.")

"""
__Block E — FitPositionsImagePairRepeatSolved (subgradient: finiteness + liveness only)__

Nearest-with-repeats pairing is a piecewise-constant selection: autodiff
returns the subgradient of the currently-selected pairing (documented on the
class), so FD steps that span the staircase can straddle a pairing flip and
disagree legitimately. The implicit rule itself is FD-certified in blocks C/D;
here we assert the solver gradients ENGAGE (finite, non-zero, live mass
parameters) without an FD sweep — every exclusion explicit, per the util.py
convention.
"""
analysis_repeat = al.AnalysisPoint(
    dataset=dataset,
    solver=solver_fine,
    fit_positions_cls=al.FitPositionsImagePairRepeatSolved,
)

fitness_repeat = Fitness(
    model=model_ip_solved,
    analysis=analysis_repeat,
    fom_is_log_likelihood=True,
    resample_figure_of_merit=-1.0e99,
)

value_rp, grad_rp = jax.value_and_grad(fitness_repeat.call)(param_vector_ip)

print(f"Log likelihood (pair-repeat solved) = {float(value_rp):.6f}")

assert np.isfinite(float(value_rp)), "Log likelihood (pair-repeat solved) is not finite"
assert np.all(
    np.isfinite(np.array(grad_rp))
), f"Gradient contains non-finite values (pair-repeat solved): {np.array(grad_rp)}"

grad_rp_arr = np.array(grad_rp)
assert np.all(
    np.abs(grad_rp_arr) > 0.0
), "A mass parameter has zero subgradient (pair-repeat solved)."

print("point_source gradient.py pair-repeat SOLVED subgradient checks passed.")

"""
__Block F — Solved fluxes + time delays (nested autodiff through the Hessian)__

``FitFluxesSolved`` needs gradients of the magnifications, i.e. third
derivatives of the deflection potential — the case the gravity.jl paper
declares out of reach analytically; nested JAX autodiff supplies them
mechanically. No solver in this chain (fluxes/delays are evaluated at the
observed positions), so the standard FD conventions apply. H0 is LIVE here:
time delays are directly H0-sensitive.
"""
dataset_ftd_path = dataset_path / "point_dataset_with_fluxes_and_time_delays.json"

if dataset_ftd_path.exists():
    dataset_ftd = al.from_json(file_path=dataset_ftd_path)

    analysis_ftd = al.AnalysisPoint(
        dataset=dataset_ftd,
        solver=solver,
        fit_positions_cls=al.FitPositionsSourceSolved,
        fit_flux_cls=al.FitFluxesSolved,
        fit_time_delays_cls=al.FitTimeDelaysSolved,
    )

    fitness_ftd = Fitness(
        model=model_solved,
        analysis=analysis_ftd,
        fom_is_log_likelihood=True,
        resample_figure_of_merit=-1.0e99,
    )

    param_vector_ftd = jnp.array(model_solved.physical_values_from_prior_medians)
    key_ftd = jax.random.PRNGKey(46)
    param_vector_ftd = param_vector_ftd + jax.random.uniform(
        key_ftd, shape=param_vector_ftd.shape, minval=0.001, maxval=0.005
    )

    value_ftd, grad_ftd = jax.value_and_grad(fitness_ftd.call)(param_vector_ftd)

    print(f"Log likelihood (fluxes+delays solved) = {float(value_ftd):.6f}")

    assert np.isfinite(float(value_ftd)), "Log likelihood (fluxes+delays) is not finite"
    assert np.all(
        np.isfinite(np.array(grad_ftd))
    ), f"Gradient contains non-finite values (fluxes+delays): {np.array(grad_ftd)}"
    assert not np.all(np.array(grad_ftd) == 0.0), "Gradient is all zeros (fluxes+delays)"

    comparison_ftd = util.compare_gradients(
        fitness_ftd.call,
        param_vector_ftd,
        param_names=util.parameter_names_from(model_solved),
    )

    util.assert_gradients_match(comparison_ftd)

    # Every parameter is live: mass moves positions/magnifications/delays, H0 the delays.
    assert np.all(
        np.abs(comparison_ftd["ad"]) > 0.0
    ), "A parameter has zero gradient in the fluxes+time-delays fit."

    print("point_source gradient.py fluxes+time-delays nested-autodiff checks passed.")
else:
    print(
        f"SKIP fluxes+time-delays gradient block: {dataset_ftd_path} not found "
        "(run scripts/point_source/simulators/simple.py to create it)."
    )

print("point_source gradient.py ALL phase-5 gradient checks passed.")
