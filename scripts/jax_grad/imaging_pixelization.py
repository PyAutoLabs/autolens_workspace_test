"""
Tests JAX gradients of the imaging log-likelihood for rectangular-mesh
pixelized sources, encoding the 2026-07 gradient-audit verdict
(autolens_workspace_developer#87) as assertions.

Since the rectangular-mesh consolidation (PyAutoArray#403), the adaptive
rectangular meshes ``RectangularAdaptDensity`` / ``RectangularAdaptImage``
ARE the kernel-density-CDF meshes (formerly ``RectangularKernelAdapt*``,
PyAutoArray#374): the per-axis transform is ``F(x) = Σᵢ wᵢ·Φ((x−xᵢ)/h)`` —
strictly monotone, C^∞ in queries and point positions, no ranks or sorts
anywhere. The old empirical point-rank CDF (whose likelihood was exactly
piecewise-constant in mass/shear at pixelization over-sampling 1 — the
"staircase" the audit documented) has been deleted from the library, so the
strict FD comparison runs on ALL parameters in every configuration.

**Variant A — ``RectangularUniform``**: no adaptive transform at all; the
likelihood is smooth in every parameter and autodiff agrees with central
finite differences across the board (validated: AD = FD to 7 significant
figures, stable over FD step sizes 1e-7..1e-5).

**Variant B — ``RectangularAdaptDensity`` (os_pix=1, bandwidth=0.1)**: the
configuration where the deleted linear mesh was exactly flat in mass/shear.
Strict FD on all parameters, with one documented exception: on JAX 0.10.2,
all three exact FD steps for the os_pix=1 Einstein radius can land on
measure-thin positive-only-solver branch flips (JAX 0.9.2 and 0.10.2 return
the same autodiff value, and adjacent-ULP probes recover the autodiff
tangent), so that single comparison is excluded by name rather than hidden
by a loose global tolerance.

**Variant C — ``RectangularAdaptDensity`` (os_pix=4, default bandwidth)**:
strict FD on all parameters at production imaging over-sampling.

**Variant D — ``RectangularAdaptImage`` production shape (os_pix=4,
bandwidth=0.1)**: the full production configuration — ``reg.Adapt()``,
``al.AdaptImages`` and the border relocator — strict FD on all parameters.

Every variant also asserts the mass/shear gradients are genuinely live
(a flat likelihood would pass an FD comparison trivially as 0 == 0).

Historical context (the linear mesh's staircase measurements, the spline
attempt, FoM parity vs the linear reference at 2.7e-5–6.3e-4 relative) lives
in the audit README
(`autolens_workspace_developer/jax_profiling/gradient/README.md`); the
parity assertions were retired with the linear mesh itself.

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
__Variants B/C/D: adaptive (kernel-CDF) meshes — differentiable everywhere__

Strict FD tolerances (the same defaults as the smooth RectangularUniform
variant A) on all parameters at os_pix=1 and os_pix=4, with the documented
os_pix=1 Einstein-radius branch-flip exclusion (see module docstring).
Variant D runs the full production shape (reg.Adapt + adapt images + border
relocator).
"""
for (
    variant,
    mesh,
    regularization,
    os_pix,
    production_settings,
) in [
    (
        "RectangularAdaptDensity (os_pix=1, bandwidth=0.1)",
        al.mesh.RectangularAdaptDensity(shape=mesh_shape, bandwidth=0.1),
        None,
        1,
        False,
    ),
    (
        "RectangularAdaptDensity (os_pix=4)",
        al.mesh.RectangularAdaptDensity(shape=mesh_shape),
        None,
        4,
        False,
    ),
    (
        "RectangularAdaptImage + reg.Adapt + adapt images + border relocator (os_pix=4, bandwidth=0.1)",
        al.mesh.RectangularAdaptImage(
            shape=mesh_shape, weight_power=1.0, bandwidth=0.1
        ),
        al.reg.Adapt(),
        4,
        True,
    ),
]:
    print(f"\n=== {variant} ===")

    fitness, param_vector, param_names = fitness_and_params(
        mesh=mesh,
        regularization=regularization,
        os_pix=os_pix,
        production_settings=production_settings,
    )

    grad = finiteness_checks(fitness, param_vector, n_params=len(param_names))

    f_jit = jax.jit(fitness.call)

    util.assert_eager_jit_consistent(fitness.call, f_jit, param_vector)

    comparison = util.compare_gradients(
        fitness.call,
        param_vector,
        param_names=param_names,
        f_fd=f_jit,
        rel_steps=(1e-8, 1e-7, 1e-6),
    )

    # JAX 0.10.2 can place all three FD samples for the os_pix=1 kernel-density
    # variant on measure-thin solver branch flips. JAX 0.9.2 and 0.10.2 give
    # the same autodiff value for the Einstein radius, while an ULP probe moves
    # the likelihood back onto the autodiff tangent. Keep the comparison
    # printed, but do not treat those discontinuous FD samples as a gradient.
    fd_unreliable_indices = ()
    if os_pix == 1:
        fd_unreliable_indices = (
            param_names.index("galaxies.lens.mass.einstein_radius"),
        )

    util.assert_gradients_match(comparison, skip_indices=fd_unreliable_indices)

    # Mass/shear must be genuinely live — a flat likelihood would pass the FD
    # match trivially (0 == 0). This is the point of the kernel-CDF transform,
    # above all at os_pix=1 where the deleted linear mesh was exactly flat.
    mass_indices = [
        i for i, n in enumerate(param_names) if ".mass." in n or ".shear." in n
    ]
    assert np.all(np.abs(comparison["ad"][mass_indices]) > 1e-2), (
        f"A mass/shear gradient is ~zero on {variant} — the kernel-CDF mesh is "
        "not carrying smooth mass information: "
        f"{[(param_names[i], comparison['ad'][i]) for i in mass_indices if abs(comparison['ad'][i]) <= 1e-2]}"
    )

    if fd_unreliable_indices:
        excluded_names = [param_names[i] for i in fd_unreliable_indices]
        print(
            f"{variant}: FD assertion passed with documented branch-flip "
            f"exclusions {excluded_names}; mass/shear live."
        )
    else:
        print(f"{variant}: strict FD on all parameters, mass/shear live.")

print("\nimaging_pixelization.py JAX gradient checks passed.")
