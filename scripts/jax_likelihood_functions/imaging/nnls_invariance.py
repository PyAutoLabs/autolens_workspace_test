"""
NNLS Invariance: Likelihood Numerics
====================================

Verifies that changes to `reconstruction_positive_only_from` (notably Jacobi
preconditioning of the jaxnnls input) do not change the log-likelihood value
produced by the MGE and Delaunay imaging pipelines for a range of parameter
vectors.

Based on `mge.py` and `delaunay.py` in this directory, but instead of a single
reference likelihood this script sweeps several perturbations of each model's
prior-median parameter vector and records the likelihood for each. The
preconditioning is a change of coordinates in the NNLS problem (``D Q D y = D q``
with ``x = D y``) that is mathematically equivalent to the unpreconditioned
solve for any feasible primal ``x``, so the final reconstruction and hence the
log-likelihood should match to within solver tolerance. Small drift (~1e-10
relative) from differing iteration counts is expected and tolerated.

Each block:
  1. Builds the model and fitness exactly as in the reference script.
  2. Computes a batch of parameter vectors: the prior-median baseline plus a
     fixed set of deterministic perturbations at different magnitudes.
  3. Evaluates `fitness.call(params)` for each vector.
  4. Asserts the numerical values match the ``BASELINE`` dict below.

Populate ``BASELINE`` from a clean (pre-change) run, then re-run after any
modification to the NNLS call path to verify numerical equivalence.
"""

import numpy as np
import jax
import jax.numpy as jnp
from os import path

import autofit as af
import autolens as al

from autofit.non_linear.fitness import Fitness


# ---------------------------------------------------------------------------
# Baseline likelihood values
#
# Populated from a clean run (before any NNLS-call-path changes). Update only
# when the solver reference changes deliberately.
# ---------------------------------------------------------------------------

BASELINE = {
    "mge": [
        -55970.52849260601,
        -59640.841986799176,
        -53768.476355086656,
        -48180.87816460625,
        -61024.79525950517,
    ],
    "delaunay": [
        -23662.84890387213,
        -22518.612284641447,
        -23441.913593155958,
        -25735.88597800103,
        -22306.16607191193,
    ],
}

RTOL = 1e-6
ATOL = 1e-4

# Fixed perturbation seeds so the parameter vectors are reproducible.
PERTURBATIONS = [
    ("baseline", 0.0, 0),
    ("small_+", 0.01, 1),
    ("small_-", -0.01, 2),
    ("medium", 0.05, 3),
    ("mixed", 0.03, 4),
]


def _perturb(vec, scale, seed):
    """Deterministic bounded perturbation of a prior-median vector."""
    if scale == 0.0:
        return jnp.array(vec)
    rng = np.random.default_rng(seed)
    mag = abs(scale)
    pert = rng.uniform(-mag, mag, size=len(vec))
    if scale < 0:
        pert = -pert
    return jnp.array(np.array(vec) + pert)


def _evaluate(fitness, vectors):
    """Call fitness.call on each parameter vector and return a numpy array."""
    values = []
    for label, vec in vectors:
        val = float(np.array(fitness.call(vec)))
        print(f"  {label:>10s}  log_likelihood = {val:.10g}")
        values.append(val)
    return np.array(values)


def _compare(key, values):
    baseline = BASELINE.get(key)
    if baseline is None:
        print(f"\n  [{key}] no baseline set -- paste this list into BASELINE:")
        print(f"    \"{key}\": {values.tolist()},")
        return
    baseline = np.array(baseline)
    abs_diff = np.abs(values - baseline)
    rel_diff = abs_diff / np.abs(baseline)
    print(f"\n  [{key}] max abs diff vs baseline: {abs_diff.max():.6g}")
    print(f"  [{key}] max rel diff vs baseline: {rel_diff.max():.6g}")
    np.testing.assert_allclose(
        values,
        baseline,
        rtol=RTOL,
        atol=ATOL,
        err_msg=f"{key}: likelihood drift exceeds tolerance (rtol={RTOL})",
    )
    print(f"  [{key}] likelihoods match baseline within rtol={RTOL}")


# ===========================================================================
# Shared dataset
# ===========================================================================

dataset_path = path.join("dataset", "imaging", "jax_test")

if al.util.dataset.should_simulate(dataset_path):
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "scripts/jax_likelihood_functions/imaging/simulator.py"],
        check=True,
    )


# ===========================================================================
# MGE model
# ===========================================================================

print("\n" + "=" * 70)
print("  MGE -- NNLS invariance sweep")
print("=" * 70)

dataset_mge = al.Imaging.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    pixel_scales=0.2,
)

mask_radius = 3.5
mask = al.Mask2D.circular(
    shape_native=dataset_mge.shape_native,
    pixel_scales=dataset_mge.pixel_scales,
    radius=mask_radius,
)
dataset_mge = dataset_mge.apply_mask(mask=mask)
dataset_mge = dataset_mge.apply_over_sampling(over_sample_size_lp=4)

over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=dataset_mge.grid,
    sub_size_list=[4, 2, 1],
    radial_list=[0.3, 0.6],
    centre_list=[(0.0, 0.0)],
)
dataset_mge = dataset_mge.apply_over_sampling(over_sample_size_lp=over_sample_size)

# Lens bulge + mass (same structure as mge.py).
bulge_lens = al.model_util.mge_model_from(
    mask_radius=mask_radius, total_gaussians=20, centre_prior_is_uniform=True
)

total_gaussians = 3
sigma_radius = 3.0
log10_sigma_list = np.linspace(-2, np.log10(sigma_radius), total_gaussians)

centre_0 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
centre_1 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)

gaussian_list = af.Collection(
    af.Model(al.lmp_linear.GaussianGradient) for _ in range(total_gaussians)
)
for i, gaussian in enumerate(gaussian_list):
    gaussian.centre.centre_0 = centre_0
    gaussian.centre.centre_1 = centre_1
    gaussian.ell_comps = gaussian_list[0].ell_comps
    gaussian.sigma = 10 ** log10_sigma_list[i]
    gaussian.mass_to_light_ratio = 10.0
    gaussian.mass_to_light_gradient = 1.0

bulge_lens = af.Model(al.lp_basis.Basis, profile_list=list(gaussian_list))
mass = af.Model(al.mp.NFWSph)
shear = af.Model(al.mp.ExternalShear)

lens = af.Model(al.Galaxy, redshift=0.5, bulge=bulge_lens, mass=mass, shear=shear)

source_bulge = al.model_util.mge_model_from(
    mask_radius=mask_radius, total_gaussians=20, centre_prior_is_uniform=False
)
source = af.Model(al.Galaxy, redshift=1.0, bulge=source_bulge)

model_mge = af.Collection(galaxies=af.Collection(lens=lens, source=source))

print(f"  free parameters: {model_mge.total_free_parameters}")

analysis_mge = al.AnalysisImaging(dataset=dataset_mge)
fitness_mge = Fitness(
    model=model_mge,
    analysis=analysis_mge,
    fom_is_log_likelihood=True,
    resample_figure_of_merit=-1.0e99,
)

medians_mge = list(model_mge.physical_values_from_prior_medians)
mge_vectors = [
    (label, _perturb(medians_mge, scale, seed))
    for (label, scale, seed) in PERTURBATIONS
]

print()
values_mge = _evaluate(fitness_mge, mge_vectors)
_compare("mge", values_mge)


# ===========================================================================
# Delaunay model
# ===========================================================================

print("\n" + "=" * 70)
print("  Delaunay -- NNLS invariance sweep")
print("=" * 70)

sub_size = 4

dataset_del = al.Imaging.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    pixel_scales=0.2,
    over_sample_size_lp=sub_size,
    over_sample_size_pixelization=sub_size,
)

mask_radius_del = 2.6
mask_del = al.Mask2D.circular(
    shape_native=dataset_del.shape_native,
    pixel_scales=dataset_del.pixel_scales,
    radius=mask_radius_del,
)
dataset_del = dataset_del.apply_mask(mask=mask_del)

over_sample_size_del = al.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=dataset_del.grid,
    sub_size_list=[4, 2, 1],
    radial_list=[0.3, 0.6],
    centre_list=[(0.0, 0.0)],
)
dataset_del = dataset_del.apply_over_sampling(
    over_sample_size_lp=over_sample_size_del,
    over_sample_size_pixelization=1,
)

pixels = 750
edge_pixels_total = 30

galaxy_image_name_dict = {
    "('galaxies', 'lens')": dataset_del.data,
    "('galaxies', 'source')": dataset_del.data,
}

image_mesh = al.image_mesh.Hilbert(pixels=pixels, weight_power=3.5, weight_floor=0.01)

image_plane_mesh_grid = image_mesh.image_plane_mesh_grid_from(
    mask=dataset_del.mask,
    adapt_data=galaxy_image_name_dict["('galaxies', 'source')"],
)
image_plane_mesh_grid = al.image_mesh.append_with_circle_edge_points(
    image_plane_mesh_grid=image_plane_mesh_grid,
    centre=mask_del.mask_centre,
    radius=mask_radius_del + mask_del.pixel_scale / 2.0,
    n_points=edge_pixels_total,
)

adapt_images = al.AdaptImages(
    galaxy_name_image_dict=galaxy_image_name_dict,
    galaxy_name_image_plane_mesh_grid_dict={
        "('galaxies', 'source')": image_plane_mesh_grid
    },
)

mass_del = af.Model(al.mp.PowerLaw)
mass_del.centre.centre_0 = af.UniformPrior(lower_limit=0.2, upper_limit=0.4)
mass_del.centre.centre_1 = af.UniformPrior(lower_limit=-0.4, upper_limit=-0.2)
mass_del.einstein_radius = af.UniformPrior(lower_limit=1.5, upper_limit=1.7)
mass_del.ell_comps.ell_comps_0 = af.UniformPrior(
    lower_limit=0.11111111111111108, upper_limit=0.1111111111111111
)
mass_del.ell_comps.ell_comps_1 = af.UniformPrior(lower_limit=-0.01, upper_limit=0.01)

shear_del = af.Model(al.mp.ExternalShear)
shear_del.gamma_1 = af.UniformPrior(lower_limit=-0.001, upper_limit=0.001)
shear_del.gamma_2 = af.UniformPrior(lower_limit=-0.001, upper_limit=0.001)

lens_del = af.Model(al.Galaxy, redshift=0.5, mass=mass_del, shear=shear_del)

regularization = al.reg.AdaptSplit()
pixelization = af.Model(
    al.Pixelization,
    mesh=al.mesh.Delaunay(pixels=pixels, zeroed_pixels=edge_pixels_total),
    regularization=regularization,
)
source_del = af.Model(al.Galaxy, redshift=1.0, pixelization=pixelization)

model_del = af.Collection(galaxies=af.Collection(lens=lens_del, source=source_del))

print(f"  free parameters: {model_del.total_free_parameters}")

analysis_del = al.AnalysisImaging(
    dataset=dataset_del,
    adapt_images=adapt_images,
    raise_inversion_positions_likelihood_exception=False,
)
fitness_del = Fitness(
    model=model_del,
    analysis=analysis_del,
    fom_is_log_likelihood=True,
    resample_figure_of_merit=-1.0e99,
)

medians_del = list(model_del.physical_values_from_prior_medians)
del_vectors = [
    (label, _perturb(medians_del, scale, seed))
    for (label, scale, seed) in PERTURBATIONS
]

print()
values_del = _evaluate(fitness_del, del_vectors)
_compare("delaunay", values_del)


# ===========================================================================
# Summary
# ===========================================================================

print("\n" + "=" * 70)
print("  NNLS invariance sweep complete")
print("=" * 70)
print("  If BASELINE is None above, paste the printed lists in and re-run.")
