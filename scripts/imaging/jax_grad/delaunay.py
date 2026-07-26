"""
Tests JAX gradients of the imaging log-likelihood for Delaunay-mesh pixelized
sources — the frozen-tables gradient unlocked 2026-07-26 (PyAutoArray: the
qhull ``pure_callback`` input is ``stop_gradient``-wrapped).

Why this yields a correct gradient: the Delaunay likelihood separates into a
DISCRETE layer (which triangle contains each query, which vertices connect —
the int32 tables the host callback returns) and a CONTINUOUS layer (vertex
positions, barycentric weights, dual areas, split points, the inversion's
linear algebra — all computed in-graph from the traced arrays). The discrete
layer is piecewise-constant in the model parameters: perturb the mass model
slightly and the triangulation is literally identical, so its true derivative
is zero everywhere except the measure-zero re-wiring (triangle-flip) events,
where the likelihood itself is discontinuous and no gradient exists for any
method. Freezing the tables under differentiation therefore drops nothing —
``jax.grad`` returns the exact almost-everywhere derivative.

The FD comparison runs the step sweep at a documented tolerance looser than
the smooth-mesh scripts: central FD steps can straddle a re-wiring event
(the interpolant jumps when a containing triangle's diagonal flips), which
contaminates individual FD samples on mass/shear parameters while autodiff
correctly differentiates the branch the evaluation point is on. Probed
2026-07-26: lens-light params match at 1e-8..1e-10; mass/shear at
1e-5..2e-3 — the scatter is FD noise at flip crossings, not autodiff error
(the same measure-zero discontinuity class as the KNN meshes' neighbour
swaps and the PyAutoArray#377 branch flips).

The model/composition mirrors ``jax_grad/knn.py`` (Hilbert image mesh +
circle edge zeroing + ``reg.AdaptSplit`` at asymmetric coefficients — the
Delaunay production shape of ``jax_likelihood/delaunay.py``, on the
truth-centred 14-parameter lens).

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

Truth-centred Gaussian priors — the same 14-parameter lens as
``jax_grad/pixelization.py`` and ``jax_grad/knn.py``.
"""

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
    mesh=al.mesh.Delaunay(pixels=pixels, zeroed_pixels=edge_pixels_total),
    regularization=al.reg.AdaptSplit(inner_coefficient=0.1, outer_coefficient=10.0),
)

source = af.Model(al.Galaxy, redshift=1.0, pixelization=pixelization)

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

"""
__Fitness__
"""
from autofit.non_linear.fitness import Fitness

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

"""
__Finiteness + FD certification__
"""
print("\n=== Delaunay + reg.AdaptSplit (frozen-tables gradient) ===")

value, grad = jax.value_and_grad(fitness.call)(param_vector)
print(f"Log likelihood = {float(value):.6f}")
assert np.isfinite(float(value)), "Log likelihood is not finite"
assert np.all(
    np.isfinite(np.array(grad))
), f"Gradient contains non-finite values: {np.array(grad)}"
assert not np.all(np.array(grad) == 0.0), "Gradient is all zeros"

f_jit = jax.jit(fitness.call)

util.assert_eager_jit_consistent(fitness.call, f_jit, param_vector)

comparison = util.compare_gradients(
    fitness.call,
    param_vector,
    param_names=param_names,
    f_fd=f_jit,
    rel_steps=(1e-8, 1e-7, 1e-6),
)

# Documented tolerance: central FD steps can straddle triangle-flip
# (re-wiring) events, contaminating individual FD samples on mass/shear
# (measured probe scatter up to ~2e-3 while lens light matches at 1e-8..1e-10;
# see module docstring). Autodiff differentiates the branch the evaluation
# point is on — a *wrong* autodiff would miss at every parameter, not sit at
# the flip-crossing noise floor.
util.assert_gradients_match(comparison, rtol=1e-2)

# Mass/shear must be genuinely live — a flat likelihood would pass the FD
# match trivially (0 == 0).
mass_indices = [
    i for i, n in enumerate(param_names) if ".mass." in n or ".shear." in n
]
assert np.all(np.abs(comparison["ad"][mass_indices]) > 1e-2), (
    "A mass/shear gradient is ~zero on the Delaunay mesh: "
    f"{[(param_names[i], comparison['ad'][i]) for i in mass_indices if abs(comparison['ad'][i]) <= 1e-2]}"
)

print(
    "Delaunay + reg.AdaptSplit: frozen-tables gradients FD-certified "
    "(rtol=1e-2, flip-crossing FD scatter documented), mass/shear live."
)

print("\ndelaunay.py JAX gradient checks passed.")
