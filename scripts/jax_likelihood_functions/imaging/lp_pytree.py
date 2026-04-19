"""
Path A PoC: jit-wrap ``analysis.fit_from`` for parametric Sersic
================================================================

Sibling of ``mge_pytree.py`` for the simpler single-profile parametric path
(see admin_jammy/prompt/issued/fit_imaging_pytree_lp.md). Uses
``al.lp.Sersic`` (not ``lp_linear.Sersic``) per the prompt's explicit
intent — this exercises the parametric, non-linear light-profile path that
MGE does not cover.

Success criterion:
  - ``jax.jit(analysis.fit_from)(instance)`` returns a ``FitImaging`` whose
    ``log_likelihood`` is a ``jax.Array`` matching the NumPy-path scalar.
"""

from os import path

import jax
import jax.numpy as jnp
import numpy as np

import autofit as af
import autolens as al

from autofit.jax.pytrees import enable_pytrees, register_model

enable_pytrees()


"""
__Dataset__ — same on-disk dataset used by ``lp.py``.
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

mask_radius = 3.0

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=mask_radius,
)

dataset = dataset.apply_mask(mask=mask)
dataset = dataset.apply_over_sampling(over_sample_size_lp=1)


"""
__Model__ — parametric Sersic bulge + PowerLaw mass + ExternalShear lens,
parametric Sersic source. Differs from ``lp.py`` only in using ``al.lp.Sersic``
instead of ``al.lp_linear.Sersic`` (prompt requires non-linear path).
"""
bulge = af.Model(al.lp.Sersic)
mass = af.Model(al.mp.PowerLaw)
shear = af.Model(al.mp.ExternalShear)

lens = af.Model(
    al.Galaxy,
    redshift=0.5,
    bulge=bulge,
    mass=mass,
    shear=shear,
)

source_bulge = af.Model(al.lp.Sersic)
source = af.Model(al.Galaxy, redshift=1.0, bulge=source_bulge)

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))


"""
__Analysis__ on the JAX path (``use_jax=True``) so ``fit_from`` builds with ``xp=jnp``.
"""
analysis = al.AnalysisImaging(dataset=dataset, use_jax=True)

register_model(model)

instance = model.instance_from_prior_medians()


"""
__NumPy reference scalar__ (cross-check once the JIT path returns).
"""
analysis_np = al.AnalysisImaging(dataset=dataset, use_jax=False)
fit_np = analysis_np.fit_from(instance=instance)
print("NumPy fit.log_likelihood:", float(fit_np.log_likelihood))


"""
__Path A: jit-wrap ``analysis.fit_from``__
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
