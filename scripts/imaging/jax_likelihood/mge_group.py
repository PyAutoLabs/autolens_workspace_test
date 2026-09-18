"""
Func Grad: Light Parametric Operated
====================================

This script test if JAX can successfully compute the gradient of the log likelihood of an `Imaging` dataset with a
model which uses operated light profiles.

 __Operated Fitting__

It is common for galaxies to have point-source emission, for example bright emission right at their centre due to
an active galactic nuclei or very compact knot of star formation.

This point-source emission is subject to blurring during data accquisiton due to the telescope optics, and therefore
is not seen as a single pixel of light but spread over multiple pixels as a convolution with the telescope
Point Spread Function (PSF).

It is difficult to model this compact point source emission using a point-source light profile (or an extremely
compact Gaussian / Sersic profile). This is because when the model-image of a compact point source of light is
convolved with the PSF, the solution to this convolution is extremely sensitive to which pixel (and sub-pixel) the
compact model emission lands in.

Operated light profiles offer an alternative approach, whereby the light profile is assumed to have already been
convolved with the PSF. This operated light profile is then fitted directly to the point-source emission, which as
discussed above shows the PSF features.

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
from jax import grad
from os import path

import autofit as af
import autolens as al
from autolens import conf


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
__Group Centres__
"""
centre_list = [(0.0, 0.0), (0.0, 1.0), (0.0, 2.0), (0.0, 3.0), (0.0, 4.0)]

"""
__Mask__

The model-fit requires a 2D mask defining the regions of the image we fit the model to the data, which we define
and use to set up the `Imaging` object that the model fits.
"""
mask = al.Mask2D.circular(
    shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=4.0
)

dataset = dataset.apply_mask(mask=mask)

# dataset = dataset.apply_over_sampling(over_sample_size_lp=4)

over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=dataset.grid,
    sub_size_list=[4, 2, 2],
    radial_list=[0.3, 0.6],
    centre_list=[(0.0, 0.0)] + centre_list,
)

dataset = dataset.apply_over_sampling(over_sample_size_lp=over_sample_size)


"""
__Model__

We compose our model using `Model` objects, which represent the galaxies we fit to our data. In this 
example we fit a model where:

 - The galaxy's bulge is a parametric `Sersic` bulge [7 parameters]. 
 - The galaxy's point source emission is a parametric operated `Gaussian` centred on the bulge [4 parameters].

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=11.

__Model Anchoring__

Every prior below whose median matters is anchored on the corresponding truth in
`scripts/imaging/simulator/simple.py`, and none of them is fixed — the free-parameter count, and
therefore the vmap/jit shape this script tests, is unchanged.

The reason is the positive-only solver. This script evaluates the likelihood at the model's prior
medians, and the inversion there solves 140 non-negative linear components (60 lens Gaussians, 30
source Gaussians, 5 x 10 extra-galaxy Gaussians) against 556 masked image pixels. When the median
model is far from the data, the lens-plane bases absorb the arcs and the solver returns the entire
source block as exactly zero — at which point the likelihood carries no source-plane mass
information at all and the mass-sensitivity assertion below cannot fire, however large the
perturbation (audit 2026-08-06, autolens_workspace_test#253; recurrence after the dataset moved to
100x100 @ 0.3", autolens_workspace_test#299).

Anchored here, with the measured effect of each:

 - the main mass at einstein_radius=1.6 (#253),
 - the lens MGE `ell_comps` on the simulator's bulge and disk, and the source MGE `ell_comps` and
   centre on the simulator's source, so the median model actually fits: chi-squared 562 on 556
   pixels, against 5650 before,
 - the extra-galaxy masses near their truth of zero (`simple.py` simulates no group members).

With all four anchored the median fit retains the source basis and a +5% mass perturbation moves the
likelihood by ~29, monotonically across satellite einstein_radius ceilings 0.01"-0.15" (19-115).
Un-anchor any one of them and the source block collapses back to zero.
"""
# Lens:

total_gaussians = 30
gaussian_per_basis = 2

# The sigma values of the Gaussians will be fixed to values spanning 0.01 to the mask radius, 3.0".
mask_radius = 3.0
log10_sigma_list = np.linspace(-2, np.log10(mask_radius), total_gaussians)

# By defining the centre here, it creates two free parameters that are assigned below to all Gaussians.

centre_0 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
centre_1 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)

bulge_gaussian_list = []

for j in range(gaussian_per_basis):
    # A list of Gaussian model components whose parameters are customized belows.

    gaussian_list = af.Collection(
        af.Model(al.lp_linear.Gaussian) for _ in range(total_gaussians)
    )

    # The two Gaussian groups stand in for the simulator's lens bulge (axis_ratio=0.9,
    # angle=45.0) and disk (axis_ratio=0.7, angle=30.0). Their `ell_comps` priors are anchored
    # on those truths rather than left at the config default (whose median is circular): at the
    # circular median the lens MGE cannot reproduce the data and the fit is dominated by that
    # mismatch, which is what starves the source basis (see __Model Anchoring__ above).

    axis_ratio, angle = [(0.9, 45.0), (0.7, 30.0)][j]
    ell_comps_truth = al.convert.ell_comps_from(axis_ratio=axis_ratio, angle=angle)

    gaussian_list[0].ell_comps.ell_comps_0 = af.UniformPrior(
        lower_limit=ell_comps_truth[0] - 0.05, upper_limit=ell_comps_truth[0] + 0.05
    )
    gaussian_list[0].ell_comps.ell_comps_1 = af.UniformPrior(
        lower_limit=ell_comps_truth[1] - 0.05, upper_limit=ell_comps_truth[1] + 0.05
    )

    # Iterate over every Gaussian and customize its parameters.

    for i, gaussian in enumerate(gaussian_list):
        gaussian.centre.centre_0 = centre_0  # All Gaussians have same y centre.
        gaussian.centre.centre_1 = centre_1  # All Gaussians have same x centre.
        gaussian.ell_comps = gaussian_list[
            0
        ].ell_comps  # All Gaussians have same elliptical components.
        gaussian.sigma = (
            10 ** log10_sigma_list[i]
        )  # All Gaussian sigmas are fixed to values above.

    bulge_gaussian_list += gaussian_list

# The Basis object groups many light profiles together into a single model component.

bulge = af.Model(
    al.lp_basis.Basis,
    profile_list=bulge_gaussian_list,
)

# The main mass is anchored near the `simulator/simple.py` truth (einstein_radius=1.6) with a
# prior whose median is that truth. At the config-default prior median (einstein_radius=4.0) the
# positive-only solver zeroes the source's solved intensities and the vmap literal below is
# bit-identical for ANY source-plane mass structure (audit 2026-08-06, autolens_workspace_test#253)
# — the mass-sensitivity assertion after the literal only works on a retained source.

mass = af.Model(al.mp.Isothermal)
mass.centre = (0.0, 0.0)
mass.ell_comps = al.convert.ell_comps_from(axis_ratio=0.8, angle=45.0)
mass.einstein_radius = af.UniformPrior(lower_limit=1.1, upper_limit=2.1)

shear = af.Model(al.mp.ExternalShear)

lens = af.Model(al.Galaxy, redshift=0.5, bulge=bulge, mass=mass)

# Source:

total_gaussians = 30
gaussian_per_basis = 1

# By defining the centre here, it creates two free parameters that are assigned to the source Gaussians.
# The centre is anchored on the simulator's source centre, (0.1, 0.1), not on (0.0, 0.0): the source
# basis has to sit where the arcs actually land in the source plane for the solver to retain it.

centre_0 = af.UniformPrior(lower_limit=0.0, upper_limit=0.2)
centre_1 = af.UniformPrior(lower_limit=0.0, upper_limit=0.2)

log10_sigma_list = np.linspace(-2, np.log10(1.0), total_gaussians)

bulge_gaussian_list = []

for j in range(gaussian_per_basis):
    gaussian_list = af.Collection(
        af.Model(al.lp_linear.Gaussian) for _ in range(total_gaussians)
    )

    # Anchored on the simulator's source (axis_ratio=0.8, angle=60.0), for the same reason as the
    # lens basis above.

    ell_comps_truth = al.convert.ell_comps_from(axis_ratio=0.8, angle=60.0)

    gaussian_list[0].ell_comps.ell_comps_0 = af.UniformPrior(
        lower_limit=ell_comps_truth[0] - 0.05, upper_limit=ell_comps_truth[0] + 0.05
    )
    gaussian_list[0].ell_comps.ell_comps_1 = af.UniformPrior(
        lower_limit=ell_comps_truth[1] - 0.05, upper_limit=ell_comps_truth[1] + 0.05
    )

    for i, gaussian in enumerate(gaussian_list):
        gaussian.centre.centre_0 = centre_0
        gaussian.centre.centre_1 = centre_1
        gaussian.ell_comps = gaussian_list[0].ell_comps
        gaussian.sigma = 10 ** log10_sigma_list[i]

    bulge_gaussian_list += gaussian_list

source_bulge = af.Model(
    al.lp_basis.Basis,
    profile_list=bulge_gaussian_list,
)

source = af.Model(al.Galaxy, redshift=1.0, bulge=source_bulge)

# Extra Galaxies:

extra_galaxies_list = []

for extra_galaxy_centre in centre_list:
    # Extra Galaxy Light

    total_gaussians = 10

    log10_sigma_list = np.linspace(-2, np.log10(mask_radius), total_gaussians)

    extra_galaxy_gaussian_list = []

    gaussian_list = af.Collection(
        af.Model(al.lp_linear.GaussianSph) for _ in range(total_gaussians)
    )

    for i, gaussian in enumerate(gaussian_list):
        gaussian.centre.centre_0 = extra_galaxy_centre[0]
        gaussian.centre.centre_1 = extra_galaxy_centre[1]
        gaussian.sigma = 10 ** log10_sigma_list[i]

    extra_galaxy_gaussian_list += gaussian_list

    extra_galaxy_bulge = af.Model(
        al.lp_basis.Basis, profile_list=extra_galaxy_gaussian_list
    )

    # Extra Galaxy Mass

    mass = af.Model(al.mp.IsothermalSph)

    # `simulator/simple.py` contains no group members, so the truth for every extra-galaxy mass is
    # zero. The prior is anchored near that truth for the same reason the main mass is anchored at
    # its own truth above: at the old median (einstein_radius=0.25 each) the five satellites add
    # ~1.25" of deflection, all of it on one side, which displaces the traced source plane centroid
    # to (0.0, +0.57)" and makes the positive-only solver zero the whole source basis.

    mass.centre = extra_galaxy_centre
    mass.einstein_radius = af.UniformPrior(lower_limit=0.0, upper_limit=0.02)

    extra_galaxy = af.Model(
        al.Galaxy, redshift=0.5, bulge=extra_galaxy_bulge, mass=mass
    )

    extra_galaxy.mass.centre = extra_galaxy_centre

    extra_galaxies_list.append(extra_galaxy)

extra_galaxies = af.Collection(extra_galaxies_list)


# Overall Lens Model:

# External Shear: a property of the system, not of the lens galaxy, so it is held in an
# `al.MassField` (a container like a galaxy, carrying no light) in its own `fields=` slot.
field = af.Model(al.MassField, redshift=0.5, shear=shear)

model = af.Collection(
    galaxies=af.Collection(lens=lens, source=source),
    fields=af.Collection(field=field),
    extra_galaxies=extra_galaxies,
)

"""
The `info` attribute shows the model in a readable format.
"""
# print(model.info)

"""
__Analysis__

The `AnalysisImaging` object defines the `log_likelihood_function` which will be used to determine if JAX
can compute its gradient.
"""
analysis = al.AnalysisImaging(
    dataset=dataset,
    #    positions_likelihood_list=[al.PositionsLH(threshold=0.4, positions=positions)],
    #   settings=al.Settings(use_positive_only_solver=False)
)


"""
The analysis and `log_likelihood_function` are internally wrapped into a `Fitness` class in **PyAutoFit**, which pairs
the model with likelihood.

This is the function on which JAX gradients are computed, so we create this class here.
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
    1185.535713,
    rtol=1e-4,
    err_msg="mge_group: JAX vmap likelihood mismatch",
)


"""
__Mass Sensitivity__

The literal above is evaluated at the model's prior medians, where a +5% change of
every lens mass parameter moves this likelihood by less than the literal's rtol
(audit 2026-08-06, autolens_workspace_test#253) — the literal alone would pass a
source-plane mass regression. This block pins mass sensitivity directly.

The floor is 9.0, set by the 2026-08-06 audit as its measured response divided by five. On the
model anchored above (see __Model Anchoring__) the measured response is 29.2 (median 1185.5357,
perturbed 1156.3320), so the floor still clears by a factor of three. Do not raise the floor to
the current response: it is the guard against the source basis being zeroed, not a pin on the
response itself, and it must keep firing on any model whose source is retained at all.
"""
mass_indices = [
    i
    for i, name in enumerate(model.model_component_and_parameter_names)
    if ".mass." in name
    and "centre" not in name
    and "ell_comps" not in name
    and "redshift" not in name
]
assert mass_indices, "imaging/mge_group: no mass parameters found for sensitivity check"

parameters_perturbed = np.array(model.physical_values_from_prior_medians)
for i in mass_indices:
    parameters_perturbed[i] *= 1.05

ll_median = float(np.asarray(result).ravel()[0])
ll_perturbed = float(
    np.asarray(fitness._vmap(jnp.array(parameters_perturbed[None, :]))).ravel()[0]
)
assert abs(ll_perturbed - ll_median) > 9.0, (
    f"imaging/mge_group: likelihood insensitive to a +5% lens-mass perturbation "
    f"(median={ll_median}, perturbed={ll_perturbed}) — source-plane mass pipeline regression?"
)
print(
    f"PASS: mass-sensitivity floor exceeded (|delta| = {abs(ll_perturbed - ll_median):.4f} > 9.0)."
)


"""
__Path A: jit-wrap ``analysis.fit_from``__
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
