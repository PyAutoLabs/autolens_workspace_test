"""
JAX Likelihood: Multipole Light Profile (Strong Lens)
======================================================

Verify that JAX can compute the log-likelihood of an ``Imaging`` fit for a
strong-lens model whose lens-galaxy bulge is a ``SersicMultipole`` with m=3
and m=4 Fourier perturbations on the eccentric radius. Two paths are exercised:

1. ``fitness._vmap`` batch evaluation (``jax.vmap`` + ``jax.jit`` on the
   autofit ``Fitness`` wrapper).
2. ``jax.jit(analysis.fit_from)`` round-trip, asserting the JIT scalar
   matches the NumPy-path scalar.

Mirrors ``imaging/lp.py`` but swaps the lens bulge for
``al.lp_linear.SersicMultipole`` and sets explicit ``TuplePrior`` Gaussian
priors on the four multipole-component parameters (the library does not yet
ship default priors for them — this keeps the script self-contained).
"""
# ENV: jax full_datasets
# JAX likelihood functions test JIT compilation; need JAX enabled
# and full-size datasets.

import numpy as np
import jax.numpy as jnp
import jax
from os import path

import autofit as af
import autolens as al


"""
__Dataset__

Load the strong-lens imaging dataset; auto-simulate if missing.
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

positions = al.Grid2DIrregular(
    al.from_json(file_path=path.join(dataset_path, "positions.json"))
)

"""
__Model__

Strong-lens model:

 - Lens bulge: linear ``SersicMultipole`` (Sersic profile with m=3, m=4 Fourier
   perturbations; intensity solved via inversion).
 - Lens mass: ``PowerLaw`` + ``ExternalShear``.
 - Source bulge: linear ``Sersic``.

The four multipole component parameters get ``TuplePrior`` Gaussian priors
centred at zero.
"""

bulge = af.Model(al.lp_linear.SersicMultipole)
bulge.multipole_3_comps = af.TuplePrior(
    multipole_3_comps_0=af.GaussianPrior(mean=0.0, sigma=0.05),
    multipole_3_comps_1=af.GaussianPrior(mean=0.0, sigma=0.05),
)
bulge.multipole_4_comps = af.TuplePrior(
    multipole_4_comps_0=af.GaussianPrior(mean=0.0, sigma=0.05),
    multipole_4_comps_1=af.GaussianPrior(mean=0.0, sigma=0.05),
)

mass = af.Model(al.mp.PowerLaw)
shear = af.Model(al.mp.ExternalShear)

lens = af.Model(
    al.Galaxy,
    redshift=0.5,
    bulge=bulge,
    mass=mass,
    shear=shear,
)

source_bulge = af.Model(al.lp_linear.Sersic)
source = af.Model(al.Galaxy, redshift=1.0, bulge=source_bulge)

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

print(model.info)

"""
__Analysis__
"""
analysis = al.AnalysisImaging(
    dataset=dataset,
    positions_likelihood_list=[al.PositionsLH(threshold=0.4, positions=positions)],
)

"""
__vmap Path__
"""
from autofit.non_linear.fitness import Fitness
import time

batch_size = 50

fitness = Fitness(
    model=model,
    analysis=analysis,
    fom_is_log_likelihood=True,
    resample_figure_of_merit=-1.0e99,
)

parameters = np.zeros((batch_size, model.total_free_parameters))
for i in range(batch_size):
    parameters[i, :] = model.physical_values_from_prior_medians
parameters = jnp.array(parameters)

start = time.time()
print()
print(fitness._vmap(parameters))
print("JAX Time To VMAP + JIT Function", time.time() - start)

start = time.time()
print()
result = fitness._vmap(parameters)
print(result)
print("JAX Time Taken using VMAP:", time.time() - start)
print("JAX Time Taken per Likelihood:", (time.time() - start) / batch_size)

"""
__Path A: jit-wrap ``analysis.fit_from``__
"""


instance = model.instance_from_prior_medians()

analysis_np = al.AnalysisImaging(
    dataset=dataset,
    positions_likelihood_list=[al.PositionsLH(threshold=0.4, positions=positions)],
    use_jax=False,
)
fit_np = analysis_np.fit_from(instance=instance)
print("NumPy fit.log_likelihood:", float(fit_np.log_likelihood))

analysis_jit = al.AnalysisImaging(
    dataset=dataset,
    positions_likelihood_list=[al.PositionsLH(threshold=0.4, positions=positions)],
    use_jax=True,
)
fit_jit_fn = jax.jit(analysis_jit.fit_from)
fit = fit_jit_fn(instance)

print("JIT fit.log_likelihood:", fit.log_likelihood)
assert isinstance(
    fit.log_likelihood, jnp.ndarray
), f"expected jax.Array, got {type(fit.log_likelihood)}"
np.testing.assert_allclose(
    float(fit.log_likelihood), float(fit_np.log_likelihood), rtol=1e-4
)
print("PASS: jit(fit_from) round-trip matches NumPy scalar.")
