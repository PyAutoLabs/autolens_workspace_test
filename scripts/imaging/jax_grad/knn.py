"""
Tests JAX gradients of the imaging log-likelihood for k-nearest-neighbour
(KNN) mesh pixelized sources — the JAX-native members of the Delaunay mesh
family (``autoarray/inversion/mesh/interpolator/knn.py``).

Both KNN meshes place their vertices with an image-mesh (Hilbert here, as in
``jax_likelihood/delaunay.py``) and trace them to the source plane, so —
unlike the Delaunay mesh itself, whose interpolation runs through a
``scipy.spatial.Delaunay`` ``pure_callback`` with no JVP rule — every stage
of their interpolation is pure JAX and the likelihood carries mass/shear
gradients through BOTH the traced query points and the traced mesh vertices:

**Variant A — ``KNearestNeighbor`` + ``reg.ConstantSplit``**: Wendland-C4
kernel weights over the k nearest mesh vertices (brute-force ``lax.top_k``
over blocks). Within a fixed neighbour set the weights are smooth; at a
neighbour swap the value is continuous (the swap happens at equal distance,
where the two candidates carry equal Wendland weight) with a measure-zero
gradient kink — the same almost-everywhere-correct structure the audit
certified for the rectangular meshes' bilinear cells.

**Variant B — ``KNearestNeighbor`` + ``reg.AdaptSplit(inner=0.1, outer=10)``**:
the adaptive split scheme with genuinely non-uniform weights (the defaults
``inner == outer == 1.0`` reduce AdaptSplit to ConstantSplit exactly), so the
pixel-signal weighting path is exercised under autodiff too.

**Variant C — ``KNNBarycentric`` + ``reg.ConstantSplit``**: locally-exact
barycentric weights on the triangle of the 3 nearest vertices. NOTE the mesh
itself FAILED its science gate as a Delaunay replacement (PyAutoArray#317:
log-evidence drifts 2.2% because ~5% of vertices are never any query's
nearest-3) — this variant certifies its *gradients*, not its science.
Unlike Wendland, a 3-nearest-set swap moves the weights discontinuously, so
its likelihood has more (still measure-zero) FD-hostile jump sites.

**Regularization constraint (asserted below)**: the neighbour-based schemes
``reg.Constant`` / ``reg.ConstantZeroth`` / ``reg.Adapt`` call
``scipy.spatial.Delaunay`` on the traced source-plane mesh grid
(``MeshGeometryDelaunay.neighbors``) — not a ``pure_callback``, a direct
numpy call — so under ``jax.jit`` / ``jax.grad`` they raise
``TracerArrayConversionError``. The split-family schemes used here derive
their spacing from kNN distances instead (pure JAX); kernel schemes
(``reg.MaternKernel``, needing tfp-nightly) are the other JAX-safe pairing.

Every variant asserts the mass/shear gradients are genuinely live, and the
strict FD comparison uses the step-sweep mode of ``util.compare_gradients``
(measure-thin solver branch flips poison single-step FD on pixelized
likelihoods — see PyAutoArray#377).

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
    pixel_scales=0.2,
)

mask_radius = 3.5

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=mask_radius,
)

dataset = dataset.apply_mask(mask=mask)

dataset = dataset.apply_over_sampling(
    over_sample_size_lp=4,
    over_sample_size_pixelization=1,
)

"""
__Image Mesh__

The KNN meshes are Delaunay-family: their vertices come from an image-mesh
computed once from the (static) adapt data, then get traced to the source
plane inside the likelihood. Mirrors ``jax_likelihood/delaunay.py``, with a
circle of edge points whose reconstruction values are zeroed.
"""
pixels = 300
edge_pixels_total = 30

galaxy_image_name_dict = {
    "('galaxies', 'lens')": dataset.data,
    "('galaxies', 'source')": dataset.data,
}

image_mesh = al.image_mesh.Hilbert(pixels=pixels, weight_power=3.5, weight_floor=0.01)

image_plane_mesh_grid = image_mesh.image_plane_mesh_grid_from(
    mask=dataset.mask, adapt_data=galaxy_image_name_dict["('galaxies', 'source')"]
)

image_plane_mesh_grid = al.image_mesh.append_with_circle_edge_points(
    image_plane_mesh_grid=image_plane_mesh_grid,
    centre=mask.mask_centre,
    radius=mask_radius + mask.pixel_scale / 2.0,
    n_points=edge_pixels_total,
)

adapt_images = al.AdaptImages(
    galaxy_name_image_dict=galaxy_image_name_dict,
    galaxy_name_image_plane_mesh_grid_dict={
        "('galaxies', 'source')": image_plane_mesh_grid
    },
)

"""
__Model__

Truth-centred Gaussian priors (the same 14-parameter lens as
``jax_grad/pixelization.py``) keep the evaluation point in a non-degenerate
region where every parameter has likelihood sensitivity.
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


def fitness_and_params(mesh, regularization):
    model = model_from(mesh=mesh, regularization=regularization)
    analysis = al.AnalysisImaging(
        dataset=dataset,
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


"""
__Variants A/B/C: KNN meshes — strict FD on all parameters__
"""
for variant, mesh, regularization in [
    (
        "KNearestNeighbor + reg.ConstantSplit",
        al.mesh.KNearestNeighbor(pixels=pixels, zeroed_pixels=edge_pixels_total),
        al.reg.ConstantSplit(coefficient=1.0),
    ),
    (
        "KNearestNeighbor + reg.AdaptSplit (inner=0.1, outer=10)",
        al.mesh.KNearestNeighbor(pixels=pixels, zeroed_pixels=edge_pixels_total),
        al.reg.AdaptSplit(inner_coefficient=0.1, outer_coefficient=10.0),
    ),
    (
        "KNNBarycentric + reg.ConstantSplit (gradients only — science gate FAILED, PyAutoArray#317)",
        al.mesh.KNNBarycentric(pixels=pixels, zeroed_pixels=edge_pixels_total),
        al.reg.ConstantSplit(coefficient=1.0),
    ),
]:
    print(f"\n=== {variant} ===")

    fitness, param_vector, param_names = fitness_and_params(
        mesh=mesh,
        regularization=regularization,
    )

    finiteness_checks(fitness, param_vector, n_params=len(param_names))

    f_jit = jax.jit(fitness.call)

    util.assert_eager_jit_consistent(fitness.call, f_jit, param_vector)

    comparison = util.compare_gradients(
        fitness.call,
        param_vector,
        param_names=param_names,
        f_fd=f_jit,
        rel_steps=(1e-8, 1e-7, 1e-6),
    )

    util.assert_gradients_match(comparison)

    # Mass/shear must be genuinely live — a flat likelihood would pass the FD
    # match trivially (0 == 0). The KNN gradient flows through both the traced
    # query points and the traced mesh vertices, so these must all be nonzero.
    mass_indices = [
        i for i, n in enumerate(param_names) if ".mass." in n or ".shear." in n
    ]
    assert np.all(np.abs(comparison["ad"][mass_indices]) > 1e-2), (
        f"A mass/shear gradient is ~zero on {variant}: "
        f"{[(param_names[i], comparison['ad'][i]) for i in mass_indices if abs(comparison['ad'][i]) <= 1e-2]}"
    )

    print(f"{variant}: strict FD on all parameters, mass/shear live.")

"""
__Documented boundary: neighbour-based regularization is numpy-only__

``reg.Constant`` (like ``reg.ConstantZeroth`` / ``reg.Adapt``) builds its
matrix from ``MeshGeometryDelaunay.neighbors`` — a direct
``scipy.spatial.Delaunay`` call on the traced source-plane mesh grid — so for
the Delaunay mesh family it cannot trace under ``jax.jit`` / ``jax.grad``.
Pin that boundary here so a future fix (e.g. kNN-derived neighbours) shows up
as this assertion failing.
"""
print("\n=== KNearestNeighbor + reg.Constant (expected TracerArrayConversionError) ===")

fitness, param_vector, param_names = fitness_and_params(
    mesh=al.mesh.KNearestNeighbor(pixels=pixels, zeroed_pixels=edge_pixels_total),
    regularization=al.reg.Constant(coefficient=1.0),
)

try:
    jax.grad(fitness.call)(param_vector)
except jax.errors.TracerArrayConversionError:
    print(
        "reg.Constant raises TracerArrayConversionError under jax.grad, as "
        "documented (scipy neighbors on the traced mesh grid)."
    )
else:
    raise AssertionError(
        "KNearestNeighbor + reg.Constant differentiated without error — the "
        "scipy-neighbors boundary documented in this script has moved. If "
        "neighbour computation is now JAX-native, promote reg.Constant to a "
        "certified variant above instead of this negative test."
    )

print("\nknn.py JAX gradient checks passed.")
