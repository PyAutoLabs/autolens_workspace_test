"""
Func Grad: SMBH Point Mass
==========================

This script tests if JAX can successfully compute the log likelihood of an `Imaging` dataset with a
model whose lens galaxy contains a central supermassive black hole, modelled as an `SMBH` point-mass
profile with its `mass` a free parameter.

 __SMBH Fitting__

A central black hole in the lens galaxy is modelled with the `SMBH` profile (a `PointMass` whose
Einstein radius is derived from a physical mass and the lens/source redshifts). Two regressions
previously broke this under JAX (PyAutoGalaxy#553):

- `PointMass.deflections_yx_2d_from` produced an `ArrayIrregular` wrapper that `jnp.multiply`
  rejects on the irregular PSF-evaluation grids every imaging fit uses.

- `SMBH.__init__` converted mass to Einstein radius with `np.sqrt`, which raises
  `TracerArrayConversionError` when `mass` is a free (traced) parameter.

The second bug is only reachable when `mass` is free, because the model instance is then built
inside the jit trace — which is why this script keeps `mass` free rather than fixed. The
profile-level half of this coverage lives in `misc/profiles_jit.py`.

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
import jax.numpy as jnp
import jax
from os import path

import autofit as af
import autolens as al

"""
__Dataset__

Load and plot the galaxy dataset via .fits files.
"""
dataset_path = path.join("dataset", "imaging", "jax_test")

"""
__Dataset Auto-Simulation__

If the dataset does not already exist on your system, it will be created by running the corresponding
simulator script. This ensures that all example scripts can be run without manually simulating data first.
"""
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
)


"""
__Mask__

The model-fit requires a 2D mask defining the regions of the image we fit the model to the data, which we define
and use to set up the `Imaging` object that the model fits.
"""
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

We compose our model using `Model` objects, which represent the galaxies we fit to our data. In this
example we fit a model where:

 - The lens galaxy's light (`Sersic` bulge + `Exponential` disk), `Isothermal` mass and
   `ExternalShear` are FIXED at the values of `simulator/simple.py` [0 parameters].
 - The lens galaxy hosts a central `SMBH` point mass whose `centre` and `mass` are FREE
   [3 parameters] — a free traced mass is the only configuration that exercises the
   `SMBH.__init__` mass-to-Einstein-radius conversion inside the jit trace.
 - The source galaxy's bulge is a linear parametric `Sersic` fixed at the simulated
   values [0 parameters].

The non-SMBH components are pinned to the simulation truth deliberately: at this workspace's
default prior medians (e.g. `effective_radius=15.0`, `einstein_radius=4.0`) the positive-only
linear solver zeroes the source's intensity, making the likelihood bit-identical for ANY
source-plane mass structure — a literal generated there would not pin the SMBH's deflections
at all. Anchored at truth the source is retained and the literal is sensitive to the SMBH.
"""
# Lens:

bulge = af.Model(al.lp_linear.Sersic)
bulge.centre = (0.0, 0.0)
bulge.ell_comps = al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0)
bulge.effective_radius = 0.6
bulge.sersic_index = 3.0

disk = af.Model(al.lp_linear.Exponential)
disk.centre = (0.0, 0.0)
disk.ell_comps = al.convert.ell_comps_from(axis_ratio=0.7, angle=30.0)
disk.effective_radius = 1.6

mass = af.Model(al.mp.Isothermal)
mass.centre = (0.0, 0.0)
mass.ell_comps = al.convert.ell_comps_from(axis_ratio=0.8, angle=45.0)
mass.einstein_radius = 1.6

shear = af.Model(al.mp.ExternalShear)
shear.gamma_1 = 0.001
shear.gamma_2 = 0.001

smbh = af.Model(al.mp.SMBH)
smbh.mass = af.LogUniformPrior(lower_limit=1e8, upper_limit=1e11)
smbh.redshift_object = 0.5
smbh.redshift_source = 1.0

lens = af.Model(
    al.Galaxy,
    redshift=0.5,
    bulge=bulge,
    disk=disk,
    mass=mass,
    smbh=smbh,
    shear=shear,
)

# Source:

source_bulge = af.Model(al.lp_linear.Sersic)
source_bulge.centre = (0.1, 0.1)
source_bulge.ell_comps = al.convert.ell_comps_from(axis_ratio=0.8, angle=60.0)
source_bulge.effective_radius = 1.0
source_bulge.sersic_index = 1.0

source = af.Model(al.Galaxy, redshift=1.0, bulge=source_bulge)

# Overall Lens Model:

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

"""
The `info` attribute shows the model in a readable format.
"""
print(model.info)

"""
__Analysis__

The `AnalysisImaging` object defines the `log_likelihood_function` which will be used to determine if JAX
can compute its likelihood and batch it with `vmap`.
"""
analysis = al.AnalysisImaging(
    dataset=dataset,
    positions_likelihood_list=[al.PositionsLH(threshold=0.4, positions=positions)],
)


"""
The analysis and `log_likelihood_function` are internally wrapped into a `Fitness` class in **PyAutoFit**, which pairs
the model with likelihood.

This is the function on which JAX gradients are computed, so we create this class here.
"""
from autofit.non_linear.fitness import Fitness
import time

batch_size = 10

fitness = Fitness(
    model=model,
    analysis=analysis,
    fom_is_log_likelihood=True,
    resample_figure_of_merit=-1.0e99,
)

param_vector = jnp.array(model.physical_values_from_prior_medians)

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

np.testing.assert_allclose(
    np.array(result),
    620.28888413,
    rtol=1e-4,
    err_msg="smbh: JAX vmap likelihood mismatch",
)


"""
__Path A: jit-wrap ``analysis.fit_from``__

Wrap ``analysis.fit_from`` in ``jax.jit`` and assert the returned ``FitImaging``
has a ``jax.Array`` ``log_likelihood`` that matches the NumPy-path scalar.
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
