"""
Func Grad: Interferometer Parametric Sersic Source
====================================================

This script tests if JAX can successfully compute the gradient of the log likelihood
of an `Interferometer` dataset with a model which uses a parametric Sersic source.

Mirrors `imaging/lp.py` but uses interferometer dataset loading (real_space_mask,
Interferometer.from_fits, TransformerDFT) from `interferometer/rectangular.py`.
No apply_over_sampling — interferometer does not oversample.
"""

import numpy as np
import jax
import jax.numpy as jnp
from os import path

import autofit as af
import autolens as al
from autoconf import conf

"""
__Mask__

We define the 'real_space_mask' which defines the grid the image the strong lens is
evaluated using.
"""
mask_radius = 3.0

real_space_mask = al.Mask2D.circular(
    shape_native=(256, 256),
    pixel_scales=0.1,
    radius=mask_radius,
)

"""
__Dataset__

Load the interferometer dataset from .fits files.
"""
dataset_name = "simple"
dataset_path = path.join("dataset", "interferometer", dataset_name)

"""
__Dataset Auto-Simulation__

If the dataset does not already exist on your system, it will be created by running
the corresponding simulator script. This ensures that all example scripts can be run
without manually simulating data first.
"""
if al.util.dataset.should_simulate(dataset_path):
    import subprocess
    import sys

    subprocess.run(
        [
            sys.executable,
            "scripts/jax_likelihood_functions/interferometer/simulator.py",
        ],
        check=True,
    )

dataset = al.Interferometer.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    uv_wavelengths_path=path.join(dataset_path, "uv_wavelengths.fits"),
    real_space_mask=real_space_mask,
    transformer_class=al.TransformerDFT,
)

print(f"Total Visiblities: {dataset.uv_wavelengths.shape[0]}")

"""
__Positions__
"""
positions = al.Grid2DIrregular(
    al.from_json(file_path=path.join(dataset_path, "positions.json"))
)

"""
__Over Sampling__

Interferometer does not observe galaxies in a way where over sampling is necessary,
therefore all interferometer calculations are performed without over sampling.
"""

"""
__Model__

We compose our model using `Model` objects, which represent the galaxies we fit to
our data. In this example we fit a model where:

 - The lens galaxy has an `Isothermal` mass and `ExternalShear`.
 - The source galaxy has a parametric `Sersic` light profile via `lp_linear.Sersic`.

The number of free parameters and therefore the dimensionality of non-linear parameter
space is N=12.
"""
# Lens:

bulge = af.Model(al.lp_linear.Sersic)

mass = af.Model(al.mp.PowerLaw)

shear = af.Model(al.mp.ExternalShear)

lens = af.Model(
    al.Galaxy,
    redshift=0.5,
    bulge=bulge,
    mass=mass,
    shear=shear,
)

# Source:

bulge = af.Model(al.lp_linear.Sersic)

source = af.Model(al.Galaxy, redshift=1.0, bulge=bulge)

# Overall Lens Model:

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

"""
The `info` attribute shows the model in a readable format.
"""
print(model.info)

"""
__Analysis__

The `AnalysisInterferometer` object defines the `log_likelihood_function` which will
be used to determine if JAX can compute its gradient.
"""
analysis = al.AnalysisInterferometer(
    dataset=dataset,
    positions_likelihood_list=[al.PositionsLH(threshold=0.4, positions=positions)],
)

"""
The analysis and `log_likelihood_function` are internally wrapped into a `Fitness`
class in **PyAutoFit**, which pairs the model with likelihood.

This is the function on which JAX gradients are computed, so we create this class here.
"""
from autofit.non_linear.fitness import Fitness
import time

batch_size = 1

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

np.testing.assert_allclose(
    np.array(result),
    -1.16915394e09,
    rtol=1e-4,
    err_msg="interferometer/lp: JAX vmap likelihood mismatch",
)


"""
__Path A: jit-wrap ``analysis.fit_from``__
"""
from autofit.jax.pytrees import enable_pytrees, register_model

enable_pytrees()
register_model(model)

instance = model.instance_from_prior_medians()

analysis_np = al.AnalysisInterferometer(
    dataset=dataset,
    positions_likelihood_list=[al.PositionsLH(threshold=0.4, positions=positions)],
    use_jax=False,
)
fit_np = analysis_np.fit_from(instance=instance)
print("NumPy fit.log_likelihood:", float(fit_np.log_likelihood))

analysis_jit = al.AnalysisInterferometer(
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
