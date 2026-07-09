"""
Tests JAX gradients of the imaging log-likelihood for rectangular-mesh
pixelized sources, encoding the 2026-07 gradient-audit verdict
(autolens_workspace_developer#87) as assertions:

**Variant A — ``RectangularUniform``**: the likelihood is smooth in every
parameter and autodiff agrees with central finite differences across the
board (validated: AD = FD to 7 significant figures, stable over FD step
sizes 1e-7..1e-5). The strict FD comparison runs on all parameters.

**Variant C — ``RectangularAdaptImage`` (production config)** and
**Variant D — ``RectangularAdaptDensity``, both at pixelization over-sampling 4**:
with over-sampling > 1 (how these meshes are used in production) the
interpolation queries no longer coincide with the transform knots, so sub-pixel
strain carries a genuine smooth mass signal and every parameter's gradient is
live and FD-matched. Variant C runs the full production shape: ``reg.Adapt()``,
``al.AdaptImages``, and the border relocator. Tolerances are looser (5%) than
the smooth variants because the finite differences — not autodiff — are
contaminated by micro-staircase jumps from rank re-orderings; the measured
FD-vs-step-size drift (2026-07-09) confirms FD converges toward autodiff as
h → 0. Mixed precision is deliberately off: float64 is required for FD.

**Variant B — ``RectangularAdaptDensity`` (pixelization over-sampling 1)**:
the adaptive mesh maps ray-traced points to rank space via a sort +
``jnp.interp`` CDF transform (`create_transforms` — the "ray-guided
transformed uniform grid" of arXiv:2606.30620). With pixelization
over-sampling 1 the interpolation queries coincide with the knots, so the
mapping — and therefore the likelihood — is **exactly invariant** under any
order-preserving deformation of the traced grid. Consequences this script
asserts:

 - lens *light* parameters are smooth and FD-matched (they bypass the mesh);
 - the likelihood is bit-identical under sub-reordering mass perturbations
   (the staircase plateau), so autodiff's ~zero mass/shear gradients are the
   *correct* almost-everywhere derivative — larger FD steps measure discrete
   rank-reordering jumps, not a slope.

Gradient-based mass inference is therefore impossible in this configuration —
not because autodiff is wrong, but because the discretisation destroys smooth
mass information. With pixelization over-sampling > 1 a genuine smooth mass
gradient reappears (sub-pixel strain between queries and knots) and autodiff
tracks it (AD ≈ FD(h→1e-7) within a few %, measured 2026-07-09); that
configuration is documented in the audit README rather than asserted here to
keep this script's runtime and semantics crisp.

If a future library change makes the adaptive mesh differentiable in the mass
parameters (e.g. a continuous density transform), variant B's invariance
assertion will fail loudly — update it and the audit README together.
"""

import numpy as np
import jax
import jax.numpy as jnp
from os import path

import autofit as af
import autolens as al

import util

dataset_name = "jax_test"
dataset_path = path.join("dataset", "imaging", dataset_name)

"""
__Dataset Auto-Simulation__
"""
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

dataset_masked = dataset.apply_mask(mask=mask)

"""
__Model__

Truth-centred Gaussian priors (mirroring the developer probe
``jax_profiling/gradient/imaging/pixelization.py``) keep the evaluation point in a
non-degenerate region: the jax_test simulator truth is an Isothermal at
einstein_radius=1.6 / q=0.8 / 45deg with a small shear, so at prior medians the
source arcs land on the mesh and every parameter has likelihood sensitivity.
"""


def model_from(mesh, regularization=None):
    lens_bulge = af.Model(al.lp.Sersic)
    lens_bulge.centre.centre_0 = af.GaussianPrior(mean=0.0, sigma=0.005)
    lens_bulge.centre.centre_1 = af.GaussianPrior(mean=0.0, sigma=0.005)
    bulge_ell = al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0)
    lens_bulge.ell_comps.ell_comps_0 = af.GaussianPrior(mean=bulge_ell[0], sigma=0.01)
    lens_bulge.ell_comps.ell_comps_1 = af.GaussianPrior(mean=bulge_ell[1], sigma=0.01)
    lens_bulge.intensity = af.GaussianPrior(mean=2.0, sigma=0.1)
    lens_bulge.effective_radius = af.GaussianPrior(mean=0.6, sigma=0.05)
    lens_bulge.sersic_index = af.GaussianPrior(mean=3.0, sigma=0.2)

    mass = af.Model(al.mp.Isothermal)
    mass.centre.centre_0 = af.GaussianPrior(mean=0.0, sigma=0.005)
    mass.centre.centre_1 = af.GaussianPrior(mean=0.0, sigma=0.005)
    mass.einstein_radius = af.GaussianPrior(mean=1.6, sigma=0.05)
    mass_ell = al.convert.ell_comps_from(axis_ratio=0.8, angle=45.0)
    mass.ell_comps.ell_comps_0 = af.GaussianPrior(mean=mass_ell[0], sigma=0.01)
    mass.ell_comps.ell_comps_1 = af.GaussianPrior(mean=mass_ell[1], sigma=0.01)

    shear = af.Model(al.mp.ExternalShear)
    shear.gamma_1 = af.GaussianPrior(mean=0.001, sigma=0.005)
    shear.gamma_2 = af.GaussianPrior(mean=0.001, sigma=0.005)

    lens = af.Model(
        al.Galaxy,
        redshift=0.5,
        bulge=lens_bulge,
        mass=mass,
        shear=shear,
    )

    if regularization is None:
        regularization = al.reg.Constant(coefficient=1.0)

    pixelization = al.Pixelization(
        mesh=mesh,
        regularization=regularization,
    )

    source = af.Model(al.Galaxy, redshift=1.0, pixelization=pixelization)

    return af.Collection(galaxies=af.Collection(lens=lens, source=source))


from autofit.non_linear.fitness import Fitness


def fitness_and_params(mesh, regularization=None, os_pix=1, production_settings=False):
    dataset = dataset_masked.apply_over_sampling(
        over_sample_size_lp=4,
        over_sample_size_pixelization=os_pix,
    )
    model = model_from(mesh=mesh, regularization=regularization)
    analysis_kwargs = {}
    if production_settings:
        analysis_kwargs["adapt_images"] = al.AdaptImages(
            galaxy_name_image_dict={
                "('galaxies', 'lens')": dataset.data,
                "('galaxies', 'source')": dataset.data,
            }
        )
        analysis_kwargs["settings"] = al.Settings(
            use_border_relocator=True,
            use_positive_only_solver=True,
        )
    analysis = al.AnalysisImaging(
        dataset=dataset,
        raise_inversion_positions_likelihood_exception=False,
        **analysis_kwargs,
    )
    fitness = Fitness(
        model=model,
        analysis=analysis,
        fom_is_log_likelihood=True,
        resample_figure_of_merit=-1.0e99,
    )
    param_vector = jnp.array(model.physical_values_from_prior_medians)
    # Small perturbation off the exact prior medians (avoids symmetric special
    # points, e.g. mass centre exactly on a grid point).
    key = jax.random.PRNGKey(42)
    perturbation = jax.random.uniform(
        key, shape=param_vector.shape, minval=0.001, maxval=0.005
    )
    param_vector = param_vector + perturbation
    param_names = util.parameter_names_from(model)
    return fitness, param_vector, param_names


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
__Variant A: RectangularUniform — strict FD correctness on all parameters__
"""
print("\n=== RectangularUniform (os_pix=1) ===")

mesh_shape = (28, 28)

fitness, param_vector, param_names = fitness_and_params(
    mesh=al.mesh.RectangularUniform(shape=mesh_shape)
)

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

print("RectangularUniform: autodiff matches finite differences on all parameters.")

"""
__Variant B: RectangularAdaptDensity — smooth lens light, staircase mass__
"""
print("\n=== RectangularAdaptDensity (os_pix=1) ===")

fitness, param_vector, param_names = fitness_and_params(
    mesh=al.mesh.RectangularAdaptDensity(shape=mesh_shape)
)

grad = finiteness_checks(fitness, param_vector, n_params=len(param_names))

f_jit = jax.jit(fitness.call)

util.assert_eager_jit_consistent(fitness.call, f_jit, param_vector)

light_indices = [i for i, n in enumerate(param_names) if ".bulge." in n]
mass_indices = [
    i for i, n in enumerate(param_names) if ".mass." in n or ".shear." in n
]

# Lens-light parameters bypass the adaptive mesh: strict FD comparison.
comparison = util.compare_gradients(
    fitness.call,
    param_vector,
    param_names=param_names,
    f_fd=f_jit,
)

util.assert_gradients_match(comparison, skip_indices=mass_indices)

# Mass/shear parameters: the likelihood must be *exactly* flat under
# sub-reordering perturbations (the staircase plateau) — this is why autodiff's
# ~zero gradients are correct, and it is the property that makes gradient-based
# mass inference impossible in this configuration.
base_value = float(f_jit(param_vector))
i_er = param_names.index("galaxies.lens.mass.einstein_radius")
for h in [1e-7, 1e-6]:
    for sign in [+1.0, -1.0]:
        shifted = np.array(param_vector)
        shifted[i_er] += sign * h
        shifted_value = float(f_jit(jnp.array(shifted)))
        assert shifted_value == base_value, (
            f"Likelihood changed under einstein_radius shift of {sign * h:+.0e} "
            f"({shifted_value} != {base_value}) — the adaptive mesh has become "
            "sensitive to smooth mass perturbations; re-run the full FD audit and "
            "update this script + the audit README."
        )

assert np.all(np.abs(np.array(grad)[mass_indices]) < 1e-6), (
    "Autodiff mass/shear gradients are no longer ~zero — the adaptive mesh "
    "differentiability picture has changed; re-run the full FD audit."
)

print(
    "RectangularAdaptDensity: lens-light gradients FD-matched; mass/shear "
    "staircase invariance confirmed (autodiff zero is correct)."
)

"""
__Variants C + D: adaptive meshes at production over-sampling (os_pix=4)__

The FD step is small (rel 1e-7) to stay below the rank-reordering scale, and the
tolerance (5%) reflects the residual micro-staircase contamination of the finite
differences — autodiff is the h-consistent reference here (FD drifts toward it as
h shrinks).
"""
for variant, mesh, regularization, production_settings in [
    (
        "RectangularAdaptImage + reg.Adapt + adapt images + border relocator (os_pix=4)",
        al.mesh.RectangularAdaptImage(shape=mesh_shape, weight_power=1.0),
        al.reg.Adapt(),
        True,
    ),
    (
        "RectangularAdaptDensity (os_pix=4)",
        al.mesh.RectangularAdaptDensity(shape=mesh_shape),
        None,
        False,
    ),
]:
    print(f"\n=== {variant} ===")

    fitness, param_vector, param_names = fitness_and_params(
        mesh=mesh,
        regularization=regularization,
        os_pix=4,
        production_settings=production_settings,
    )

    grad = finiteness_checks(fitness, param_vector, n_params=len(param_names))

    f_jit = jax.jit(fitness.call)

    util.assert_eager_jit_consistent(fitness.call, f_jit, param_vector)

    comparison = util.compare_gradients(
        fitness.call,
        param_vector,
        param_names=param_names,
        rel_step=1e-7,
        f_fd=f_jit,
    )

    util.assert_gradients_match(comparison, rtol=0.05, atol=1.0)

    # Every parameter — including mass and shear — must be live at production
    # over-sampling: this is the configuration gradient-based inference will use.
    assert np.all(np.abs(comparison["ad"]) > 1.0), (
        "A parameter gradient is ~zero at os_pix=4 — the adaptive mesh has lost "
        f"smooth sensitivity: {[(n, a) for n, a in zip(param_names, comparison['ad']) if abs(a) <= 1.0]}"
    )

    print(f"{variant}: all gradients live and FD-matched (5% tolerance).")

print("\nimaging_pixelization.py JAX gradient checks passed.")
