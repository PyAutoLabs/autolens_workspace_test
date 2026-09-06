"""
Tests JAX gradients of the imaging log-likelihood across REGULARIZATION
schemes, on both gradient-capable mesh families — the coverage the meshes'
own jax_grad scripts (``pixelization.py``, ``knn.py``) leave implicit.

A 2026-07-26 sweep of every ``al.reg`` scheme against the rectangular
(kernel-CDF) and k-nearest-neighbour meshes mapped the compatibility
surface; this script pins its four load-bearing positive results:

**Variant A — ``RectangularRTUAdaptDensity`` (os_pix=4) + ``reg.Zeroth``**:
zeroth-order regularization (each pixel toward zero) is neighbour-free and
pure xp — strict FD on all parameters.

**Variant B — ``RectangularRTUAdaptDensity`` (os_pix=4) +
``reg.MaternKernel(nu=2.5)``**: THE Matérn/tfp question. The JAX path of
the Matérn kernel evaluates the modified Bessel ``K_nu`` through
``tensorflow_probability.substrates.jax.math.bessel_kve`` (tfp-nightly —
see ``kv_xp`` in ``PyAutoArray .../regularization/matern_kernel.py``),
which ships a registered custom gradient w.r.t. its argument (``nu`` is a
static float), and the dense covariance is inverted via a differentiable
Cholesky ``cho_solve``. Gradients flow end-to-end and pass strict FD on all
parameters on this mesh (measured 2.2e-4 max relative).

**Variant C — ``KNearestNeighbor`` + ``reg.Zeroth``**: strict FD on all
parameters (measured 7.5e-8 max relative).

**Variant D — ``KNearestNeighbor`` + ``reg.MaternKernel(nu=2.5)``**: the
same Matérn algebra on a mesh whose vertices are TRACED (clustered) points.
The kernel covariance of clustered points is ill-conditioned —
cond(C) ~ 1e9 measured here, vs ~3e5 for the rectangular mesh's well-spaced
vertices — so forming ``coefficient * C^-1`` explicitly puts a ~1e-6
absolute numerical noise floor on the likelihood itself (visible as the
eager-vs-jit difference). Central finite differences divide that noise by
the step, so FD cannot resolve below ~a few 1e-3 relative on this variant —
the comparison therefore runs at a documented ``rtol=1e-2`` rather than the
strict default, and the sharper statement is variant B, where conditioning
is healthy and the identical code path passes strictly. (If the kernel
schemes' linear algebra is later reformulated to avoid the explicit
inverse — e.g. keeping H implicit via the Cholesky of C — this tolerance
should be re-tightened.)

The sweep's negative results are pinned in ``knn.py`` (neighbour-based
schemes raise ``TracerArrayConversionError`` on the Delaunay mesh family)
and recorded in the gradient-audit README
(``autolens_workspace_developer/jax_profiling/gradient/README.md``):
``BrightnessZeroth`` and ``ExponentialKernel`` are not yet xp-ported
(numpy ops on traced arrays), the split-family schemes are structurally
incompatible with the rectangular meshes' shared 4-corner mappings, and
``CurvatureMask`` / ``FourthOrderMask`` are potential-correction (dpsi)
schemes sized to the data grid, not source-mesh schemes.

__Env__

Test-harness configuration (PyAutoHands docs/env_profile_redesign.md §10).
Drive jax.value_and_grad + finite-difference gradient checks; need JAX
enabled and full-resolution float64 data. Requires tfp-nightly for the
Matérn variants (``pip install tfp-nightly``; the last stable
tensorflow-probability release is incompatible with modern JAX).

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

dataset_name = "jax_test"
dataset_path = path.join("dataset", "imaging", dataset_name)

"""
__Dataset Auto-Simulation__
"""
if al.util.dataset.should_simulate(dataset_path):
    import subprocess

    subprocess.run(
        [sys.executable, "scripts/imaging/simulator/simple.py"],
        check=True,
    )

dataset = al.Imaging.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    pixel_scales=0.3,
)

mask_radius = 3.5

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=mask_radius,
)

dataset_masked = dataset.apply_mask(mask=mask)

"""
__Image Mesh (KNN variants)__
"""
# Sized from the mask rather than inherited: the 3.5" mask holds 432 image pixels at this
# dataset's 0.3" pixel scale, and this script's autodiff-vs-finite-difference certification is a
# conditioning comparison — at 300 mesh points plus 30 edge points (a ratio of 0.76) it fails by
# 4% against its 1% tolerance, while 200 (a ratio of 0.53) reproduces the gradients. 130 was also
# measured and under-resolves the source badly enough to fail the same comparison far worse.
pixels = 200
edge_pixels_total = 30

galaxy_image_name_dict = {
    "('galaxies', 'lens')": dataset_masked.data,
    "('galaxies', 'source')": dataset_masked.data,
}

image_mesh = al.image_mesh.Hilbert(pixels=pixels, weight_power=3.5, weight_floor=0.01)

image_plane_mesh_grid = image_mesh.image_plane_mesh_grid_from(
    mask=dataset_masked.mask,
    adapt_data=galaxy_image_name_dict["('galaxies', 'source')"],
)

image_plane_mesh_grid = al.image_mesh.append_with_circle_edge_points(
    image_plane_mesh_grid=image_plane_mesh_grid,
    centre=mask.mask_centre,
    radius=mask_radius + mask.pixel_scale / 2.0,
    n_points=edge_pixels_total,
)

adapt_images_rect = al.AdaptImages(galaxy_name_image_dict=galaxy_image_name_dict)

adapt_images_knn = al.AdaptImages(
    galaxy_name_image_dict=galaxy_image_name_dict,
    galaxy_name_image_plane_mesh_grid_dict={
        "('galaxies', 'source')": image_plane_mesh_grid
    },
)

"""
__Model__

Truth-centred Gaussian priors — the same 14-parameter lens as
``jax_grad/pixelization.py`` and ``jax_grad/knn.py``.
"""


def model_from(mesh, regularization):
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

    pixelization = af.Model(
        al.Pixelization,
        mesh=mesh,
        regularization=regularization,
    )

    source = af.Model(al.Galaxy, redshift=1.0, pixelization=pixelization)

    return af.Collection(galaxies=af.Collection(lens=lens, source=source))


from autofit.non_linear.fitness import Fitness


def fitness_and_params(mesh, regularization, os_pix, adapt_images):
    ds = dataset_masked.apply_over_sampling(
        over_sample_size_lp=4,
        over_sample_size_pixelization=os_pix,
    )
    model = model_from(mesh=mesh, regularization=regularization)
    analysis = al.AnalysisImaging(
        dataset=ds,
        adapt_images=adapt_images,
        raise_inversion_positions_likelihood_exception=False,
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


mesh_shape = (28, 28)

for (
    variant,
    mesh,
    regularization,
    os_pix,
    adapt_images,
    rtol,
) in [
    (
        "RectangularRTUAdaptDensity (os_pix=4) + reg.Zeroth",
        al.mesh.RectangularRTUAdaptDensity(shape=mesh_shape),
        al.reg.Zeroth(coefficient=1.0),
        4,
        adapt_images_rect,
        1e-3,
    ),
    (
        "RectangularRTUAdaptDensity (os_pix=4) + reg.MaternKernel(nu=2.5) [tfp bessel_kve]",
        al.mesh.RectangularRTUAdaptDensity(shape=mesh_shape),
        al.reg.MaternKernel(coefficient=100.0, scale=1.0, nu=2.5),
        4,
        adapt_images_rect,
        1e-3,
    ),
    (
        "KNearestNeighbor + reg.Zeroth",
        al.mesh.KNearestNeighbor(pixels=pixels, zeroed_pixels=edge_pixels_total),
        al.reg.Zeroth(coefficient=1.0),
        1,
        adapt_images_knn,
        1e-3,
    ),
    (
        "KNearestNeighbor + reg.MaternKernel(nu=2.5) [conditioning-floor tolerance]",
        al.mesh.KNearestNeighbor(pixels=pixels, zeroed_pixels=edge_pixels_total),
        al.reg.MaternKernel(coefficient=100.0, scale=1.0, nu=2.5),
        1,
        adapt_images_knn,
        # cond(C) ~ 1e9 on traced (clustered) vertices puts a ~1e-6 absolute
        # noise floor on the likelihood via the explicit C^-1; FD cannot
        # resolve below ~a few 1e-3 relative here (see module docstring).
        # Variant B certifies the identical code path strictly.
        1e-2,
    ),
]:
    print(f"\n=== {variant} ===")

    fitness, param_vector, param_names = fitness_and_params(
        mesh=mesh,
        regularization=regularization,
        os_pix=os_pix,
        adapt_images=adapt_images,
    )

    finiteness_checks(fitness, param_vector, n_params=len(param_names))

    f_jit = jax.jit(fitness.call)

    # The dense-kernel inverse carries a small eager-vs-jit reduction-order
    # noise floor (rel ~1e-10..1e-9 on ill-conditioned variants); the strict
    # 1e-10 default is kept for the well-conditioned ones.
    util.assert_eager_jit_consistent(
        fitness.call, f_jit, param_vector, rtol=1e-8 if rtol > 1e-3 else 1e-10
    )

    comparison = util.compare_gradients(
        fitness.call,
        param_vector,
        param_names=param_names,
        f_fd=f_jit,
        rel_steps=(1e-8, 1e-7, 1e-6),
    )

    util.assert_gradients_match(comparison, rtol=rtol)

    # Mass/shear must be genuinely live — a flat likelihood would pass the FD
    # match trivially (0 == 0).
    mass_indices = [
        i for i, n in enumerate(param_names) if ".mass." in n or ".shear." in n
    ]
    assert np.all(np.abs(comparison["ad"][mass_indices]) > 1e-2), (
        f"A mass/shear gradient is ~zero on {variant}: "
        f"{[(param_names[i], comparison['ad'][i]) for i in mass_indices if abs(comparison['ad'][i]) <= 1e-2]}"
    )

    print(f"{variant}: FD passed at rtol={rtol}, mass/shear live.")

print("\nregularization.py JAX gradient checks passed.")
