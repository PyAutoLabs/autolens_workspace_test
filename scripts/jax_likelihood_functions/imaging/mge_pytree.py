"""
Path A PoC: jit-wrap ``analysis.fit_from`` for MGE
==================================================

Exercises Path A of the ``fit-imaging-pytree`` task (issue
https://github.com/PyAutoLabs/PyAutoLens/issues/444). We take the same MGE
parametric lens + MGE source model from ``mge.py`` and wrap
``analysis.fit_from`` in ``jax.jit`` so the returned ``FitImaging`` (and every
nested autoarray / galaxy / tracer type it carries) has to round-trip through
JAX's pytree machinery.

The first failure we hit on each run is the next type that needs pytree
registration. This script is the iteration driver for that work.

Success criteria for this PoC (not all satisfied yet — step-by-step):
  - ``jax.jit(analysis.fit_from)(instance)`` returns a ``FitImaging``.
  - ``fit.log_likelihood`` is a ``jax.Array`` matching the NumPy-path scalar.
"""

from os import path

import jax
import jax.numpy as jnp
import numpy as np

import autofit as af
import autolens as al

# Register PyAutoFit priors, Model, Collection, ModelInstance as pytrees.
from autofit.jax.pytrees import enable_pytrees, register_model

enable_pytrees()


"""
__Dataset__
"""
dataset_path = path.join("dataset", "imaging", "jax_test")

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

dataset = dataset.apply_mask(mask=mask)
dataset = dataset.apply_over_sampling(over_sample_size_lp=4)

over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=dataset.grid,
    sub_size_list=[4, 2, 1],
    radial_list=[0.3, 0.6],
    centre_list=[(0.0, 0.0)],
)
dataset = dataset.apply_over_sampling(over_sample_size_lp=over_sample_size)


"""
__Model__ (same MGE + NFWSph + ExternalShear lens, MGE source as ``mge.py``)
"""
bulge = al.model_util.mge_model_from(
    mask_radius=mask_radius, total_gaussians=20, centre_prior_is_uniform=True
)

mass = af.Model(al.mp.NFWSph)

total_gaussians = 3
mask_radius = 3.0
log10_sigma_list = np.linspace(-2, np.log10(mask_radius), total_gaussians)

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

bulge_gaussian_list = list(gaussian_list)

bulge = af.Model(
    al.lp_basis.Basis,
    profile_list=bulge_gaussian_list,
)

shear = af.Model(al.mp.ExternalShear)

lens = af.Model(al.Galaxy, redshift=0.5, bulge=bulge, mass=mass, shear=shear)

bulge_source = al.model_util.mge_model_from(
    mask_radius=mask_radius, total_gaussians=20, centre_prior_is_uniform=False
)

source = af.Model(al.Galaxy, redshift=1.0, bulge=bulge_source)

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))


"""
__Analysis__ — on the JAX path so ``fit_from`` builds with ``xp=jnp``.
"""
analysis = al.AnalysisImaging(dataset=dataset, use_jax=True)


"""
Walk the model and register every user class (``Galaxy``, light/mass profiles,
etc.) with ``jax.tree_util`` so ``ModelInstance`` can flatten to leaves.
"""
register_model(model)

instance = model.instance_from_prior_medians()


"""
__NumPy reference scalar__ (for cross-check once the JIT path returns).
"""
analysis_np = al.AnalysisImaging(dataset=dataset, use_jax=False)
fit_np = analysis_np.fit_from(instance=instance)
print("NumPy fit.log_likelihood:", float(fit_np.log_likelihood))


"""
__Path A: jit-wrap ``analysis.fit_from``__

The first run exposes whichever type in the ``FitImaging`` closure is not yet
pytree-registered. Subsequent commits register each offender until the jit
succeeds and returns a ``FitImaging`` with ``jax.Array`` leaves.
"""
fit_jit_fn = jax.jit(analysis.fit_from)
fit = fit_jit_fn(instance)

print("JIT fit type:", type(fit).__name__)
print("JIT fit.log_likelihood:", fit.log_likelihood)
assert isinstance(fit.log_likelihood, jnp.ndarray), (
    f"expected jax.Array, got {type(fit.log_likelihood)}"
)
np.testing.assert_allclose(
    float(fit.log_likelihood), float(fit_np.log_likelihood), rtol=1e-4
)

print("PASS: jit(fit_from) round-trip matches NumPy scalar.")
