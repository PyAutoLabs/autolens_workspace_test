"""
Func Grad: Multi Galaxy Light Parametric
========================================

This script tests that JAX can JIT-compile and batch-evaluate (via `jax.vmap`) the log likelihood of an
`Imaging` dataset fitted with a **multi-galaxy** model: two co-dominant lens galaxies, each with its own
free parametric light and (untruncated) mass model, plus an `ExternalShear` on the first galaxy.

The multi-galaxy regime (see `autolens_workspace/scripts/multi_galaxy/`) reuses the standard
extended-source `AnalysisImaging` pipeline — what this script locks in is that the *summed* deflection
field of N free deflectors traces and differentiates cleanly through the JAX likelihood path, exactly as
the single-deflector `imaging/jax_likelihood/lp.py` does for one.

Its siblings `multi_galaxy/model_fit.py` (real-search fit path) and `multi_galaxy/composition_mge.py`
(model composition) cover the non-JAX legs of the regime.

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
import jax.numpy as jnp

import autofit as af
import autolens as al

"""
__Simulate__

Simulate the merging-pair dataset inline (identical to `multi_galaxy/model_fit.py`): two
`SersicSph` + `IsothermalSph` galaxies of comparable Einstein radius (1.0" and 0.8") and a compact
`SersicCore` source. The fixed `noise_seed` makes the dataset — and therefore the hardcoded likelihood
literal below — deterministic.
"""
grid = al.Grid2D.uniform(shape_native=(80, 80), pixel_scales=0.2)

psf = al.Convolver.from_gaussian(
    convolve_over_sample_size=1,
    shape_native=(11, 11),
    sigma=0.2,
    pixel_scales=grid.pixel_scales,
)

simulator = al.SimulatorImaging(
    exposure_time=300.0,
    psf=psf,
    background_sky_level=0.1,
    add_poisson_noise_to_data=True,
    noise_seed=1,
)

main_lens_centres = [(0.35, 0.25), (-0.35, -0.25)]

lens_0 = al.Galaxy(
    redshift=0.5,
    bulge=al.lp.SersicSph(
        centre=main_lens_centres[0],
        intensity=1.0,
        effective_radius=0.6,
        sersic_index=3.0,
    ),
    mass=al.mp.IsothermalSph(centre=main_lens_centres[0], einstein_radius=1.0),
)

lens_1 = al.Galaxy(
    redshift=0.5,
    bulge=al.lp.SersicSph(
        centre=main_lens_centres[1],
        intensity=0.8,
        effective_radius=0.5,
        sersic_index=3.0,
    ),
    mass=al.mp.IsothermalSph(centre=main_lens_centres[1], einstein_radius=0.8),
)

source_galaxy = al.Galaxy(
    redshift=1.0,
    bulge=al.lp.SersicCore(
        centre=(0.0, 0.03), intensity=3.0, effective_radius=0.3, sersic_index=1.0
    ),
)

tracer = al.Tracer(galaxies=[lens_0, lens_1, source_galaxy])

dataset = simulator.via_tracer_from(tracer=tracer, grid=grid)

"""
__Mask__

The mask must enclose the *combined* Einstein ring of the pair (~1.8"), not just one galaxy's light.
"""
mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=3.0,
)

dataset = dataset.apply_mask(mask=mask)

over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=dataset.grid,
    sub_size_list=[4, 2, 2],
    radial_list=[0.3, 0.6],
    centre_list=main_lens_centres,
)

dataset = dataset.apply_over_sampling(over_sample_size_lp=over_sample_size)

"""
__Model__

One free light + mass model per co-dominant deflector, composed with the same list-based `lens_0`,
`lens_1`, ... API the workspace package uses. Mass profiles are **untruncated** isothermals by design
(no host halo means no tidal truncation). Only `lens_0` carries the `ExternalShear`.
"""
# The simulated values, copied from the simulation block above. Every prior below is uniform and
# centred on them, so the prior-median evaluation this model is used for sits at the truth rather
# than at the workspace defaults.
main_lens_truth = [
    {"intensity": 1.0, "effective_radius": 0.6, "sersic_index": 3.0, "einstein_radius": 1.0},
    {"intensity": 0.8, "effective_radius": 0.5, "sersic_index": 3.0, "einstein_radius": 0.8},
]

lens_dict = {}

for i, centre in enumerate(main_lens_centres):
    truth = main_lens_truth[i]

    bulge = af.Model(al.lp.SersicSph)
    bulge.centre = centre
    bulge.intensity = af.UniformPrior(
        lower_limit=0.9 * truth["intensity"], upper_limit=1.1 * truth["intensity"]
    )
    bulge.effective_radius = af.UniformPrior(
        lower_limit=truth["effective_radius"] - 0.1,
        upper_limit=truth["effective_radius"] + 0.1,
    )
    bulge.sersic_index = af.UniformPrior(
        lower_limit=truth["sersic_index"] - 0.5,
        upper_limit=truth["sersic_index"] + 0.5,
    )

    mass = af.Model(al.mp.IsothermalSph)
    mass.centre = centre
    mass.einstein_radius = af.UniformPrior(
        lower_limit=truth["einstein_radius"] - 0.1,
        upper_limit=truth["einstein_radius"] + 0.1,
    )

    shear = None

    if i == 0:
        # There is no shear in the simulated system, so both components have median zero.
        shear = af.Model(al.mp.ExternalShear)
        shear.gamma_1 = af.UniformPrior(lower_limit=-0.01, upper_limit=0.01)
        shear.gamma_2 = af.UniformPrior(lower_limit=-0.01, upper_limit=0.01)

    lens_dict[f"lens_{i}"] = af.Model(
        al.Galaxy,
        redshift=0.5,
        bulge=bulge,
        mass=mass,
        shear=shear,
    )

# The simulated source: `SersicCore(centre=(0.0, 0.03), intensity=3.0, effective_radius=0.3,
# sersic_index=1.0)` with the default (zero) `ell_comps`, whose own prior medians are already zero.
source_bulge = af.Model(al.lp.SersicCore)
source_bulge.centre.centre_0 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
source_bulge.centre.centre_1 = af.UniformPrior(lower_limit=-0.07, upper_limit=0.13)
source_bulge.intensity = af.UniformPrior(lower_limit=2.7, upper_limit=3.3)
source_bulge.effective_radius = af.UniformPrior(lower_limit=0.25, upper_limit=0.35)
source_bulge.sersic_index = af.UniformPrior(lower_limit=0.9, upper_limit=1.1)

source = af.Model(al.Galaxy, redshift=1.0, bulge=source_bulge)

model = af.Collection(galaxies=af.Collection(**lens_dict, source=source))

"""
The `info` attribute shows the model in a readable format.
"""
print(model.info)

"""
__Analysis__

The `AnalysisImaging` object defines the `log_likelihood_function` which will be used to determine if JAX
can compute its gradient.
"""
analysis = al.AnalysisImaging(
    dataset=dataset,
)

"""
The analysis and `log_likelihood_function` are internally wrapped into a `Fitness` class in **PyAutoFit**,
which pairs the model with likelihood.

This is the function on which JAX gradients are computed, so we create this class here.
"""
from autofit.non_linear.fitness import Fitness
import time

batch_size = 3

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
    1206.80145178,
    rtol=1e-4,
    err_msg="multi_galaxy lp: JAX vmap likelihood mismatch",
)

"""
__Path A: jit-wrap ``analysis.fit_from``__

Wrap ``analysis.fit_from`` in ``jax.jit`` and assert the returned ``FitImaging``
has a ``jax.Array`` ``log_likelihood`` that matches the NumPy-path scalar.
"""

instance = model.instance_from_prior_medians()

analysis_np = al.AnalysisImaging(dataset=dataset, use_jax=False)
fit_np = analysis_np.fit_from(instance=instance)
print("NumPy fit.log_likelihood:", float(fit_np.log_likelihood))

analysis_jit = al.AnalysisImaging(dataset=dataset, use_jax=True)
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
