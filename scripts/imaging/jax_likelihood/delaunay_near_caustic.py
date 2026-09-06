"""
JAX Likelihood Parity: Delaunay Near-Caustic (likelihood-invariance guard)
==========================================================================

Parity guard for the qhull-only JAX Delaunay callback (PyAutoArray#367 /
PR #368): the JAX likelihood must match the eager NumPy reference — which
still uses scipy's find_simplex directly and is therefore the numerical
ground truth — to fp precision, on a configuration chosen to stress the
point locator where it is hardest:

- High-ellipticity PowerLaw + strong external shear, so the source-plane
  caustic is extended and the traced Delaunay mesh contains folded, highly
  compressed sliver triangles.
- ConstantSplit regularization, so the split-cross point location (the
  second locator call) is exercised too.
- Border relocation on, so hull-edge / outside-hull fallback paths are hit
  by a large fraction of the traced data grid.

Asserts eager (xp=np) == jit == every vmap lane at rtol=1e-8 — far tighter
than the 1e-4 of the sibling scripts, because the point of PR #368 is that
the mappings are EXACT (visibility walk), so the only residual is fp
reduction ordering (~1e-13 relative, measured).

__Env__

Test-harness configuration (PyAutoHands docs/env_profile_redesign.md §10).
JAX likelihood functions test JIT compilation; need JAX enabled and full-
size datasets.

ENV: jax full_datasets
"""

# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import numpy as np
import jax
from os import path

import autofit as af
import autolens as al

sub_size = 4

"""
__Dataset__
"""
dataset_path = path.join("dataset", "imaging", "jax_test")

if al.util.dataset.should_simulate(dataset_path):
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "scripts/imaging/simulator/simple.py"],
        check=True,
    )

dataset = al.Imaging.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    pixel_scales=0.3,
    over_sample_size_lp=sub_size,
    over_sample_size_pixelization=sub_size,
)

mask_radius = 2.6

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=mask_radius,
)

dataset = dataset.apply_mask(mask=mask)

over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=dataset.grid,
    sub_size_list=[4, 2, 2],
    radial_list=[0.3, 0.6],
    centre_list=[(0.0, 0.0)],
)

dataset = dataset.apply_over_sampling(
    over_sample_size_lp=over_sample_size,
    over_sample_size_pixelization=1,
)

"""
__Mesh__
"""
pixels = 200

galaxy_image_name_dict = {
    "('galaxies', 'lens')": dataset.data,
    "('galaxies', 'source')": dataset.data,
}

image_mesh = al.image_mesh.Hilbert(pixels=pixels, weight_power=3.5, weight_floor=0.01)

image_plane_mesh_grid = image_mesh.image_plane_mesh_grid_from(
    mask=dataset.mask, adapt_data=galaxy_image_name_dict["('galaxies', 'source')"]
)

adapt_images = al.AdaptImages(
    galaxy_name_image_dict=galaxy_image_name_dict,
    galaxy_name_image_plane_mesh_grid_dict={
        "('galaxies', 'source')": image_plane_mesh_grid
    },
)

"""
__Model__

Near-caustic stress configuration: the priors pin an extreme axis-ratio
PowerLaw with strong shear at the prior medians, producing an extended,
folded caustic through the source mesh.
"""
mass = af.Model(al.mp.PowerLaw)

mass.centre.centre_0 = af.UniformPrior(lower_limit=0.29, upper_limit=0.31)
mass.centre.centre_1 = af.UniformPrior(lower_limit=-0.31, upper_limit=-0.29)
mass.einstein_radius = af.UniformPrior(lower_limit=1.58, upper_limit=1.62)
# axis ratio ~0.5 at 45 deg — much flatter than the sibling script
mass.ell_comps.ell_comps_0 = af.UniformPrior(lower_limit=0.32, upper_limit=0.34)
mass.ell_comps.ell_comps_1 = af.UniformPrior(lower_limit=-0.01, upper_limit=0.01)
mass.slope = af.UniformPrior(lower_limit=2.09, upper_limit=2.11)

shear = af.Model(al.mp.ExternalShear)
shear.gamma_1 = af.UniformPrior(lower_limit=0.09, upper_limit=0.11)
shear.gamma_2 = af.UniformPrior(lower_limit=0.04, upper_limit=0.06)

lens = af.Model(al.Galaxy, redshift=0.5, mass=mass, shear=shear)

pixelization = af.Model(
    al.Pixelization,
    mesh=al.mesh.Delaunay(pixels=pixels, zeroed_pixels=0),
    regularization=al.reg.ConstantSplit(coefficient=1.0),
)

source = af.Model(al.Galaxy, redshift=1.0, pixelization=pixelization)

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

# register the model's pytrees so a ModelInstance can cross the jax.jit
# boundary (the sibling scripts get this implicitly from constructing
# Fitness first; here the jit round-trip runs first)
from autofit.jax import register_model

register_model(model)

print(model.info)

"""
__Eager NumPy reference (the ground truth: scipy find_simplex path)__
"""
import jax.numpy as jnp

instance = model.instance_from_prior_medians()

analysis_np = al.AnalysisImaging(
    dataset=dataset,
    adapt_images=adapt_images,
    raise_inversion_positions_likelihood_exception=False,
    use_jax=False,
)
fit_np = analysis_np.fit_from(instance=instance)
log_likelihood_np = float(fit_np.log_likelihood)
figure_of_merit_np = float(fit_np.figure_of_merit)
print("NumPy fit.log_likelihood:", log_likelihood_np)
print("NumPy fit.figure_of_merit:", figure_of_merit_np)

"""
__JIT round-trip__
"""
analysis_jit = al.AnalysisImaging(
    dataset=dataset,
    adapt_images=adapt_images,
    raise_inversion_positions_likelihood_exception=False,
    use_jax=True,
)
fit_jit_fn = jax.jit(analysis_jit.fit_from)
fit = fit_jit_fn(instance)

print("JIT fit.log_likelihood:", fit.log_likelihood)
np.testing.assert_allclose(
    float(fit.log_likelihood),
    log_likelihood_np,
    rtol=1e-8,
    err_msg=(
        "delaunay_near_caustic: JIT likelihood diverged from the eager scipy "
        "reference — the JAX point locator no longer reproduces find_simplex "
        "(likelihood-invariance contract of PyAutoArray#367)"
    ),
)
print("PASS: jit(fit_from) matches the eager scipy reference at rtol=1e-8.")

"""
__vmap consistency (every lane must equal the jit value)__
"""
from autofit.non_linear.fitness import Fitness

batch_size = 2

fitness = Fitness(
    model=model,
    analysis=analysis_jit,
    fom_is_log_likelihood=True,
    resample_figure_of_merit=-1.0e99,
    batch_size=batch_size,
)

parameters = np.tile(
    np.asarray(model.physical_values_from_prior_medians), (fitness.batch_size, 1)
)
result = fitness._vmap(jnp.array(parameters))
print("vmap batch:", result)

# Fitness returns the figure of merit (log evidence), NOT fit.log_likelihood
# — compare like with like (the sibling scripts pin this quantity too).
np.testing.assert_allclose(
    np.array(result),
    figure_of_merit_np,
    rtol=1e-8,
    err_msg="delaunay_near_caustic: vmap lanes diverged from the eager reference",
)
print("PASS: vmap lanes match the eager scipy figure_of_merit at rtol=1e-8.")
