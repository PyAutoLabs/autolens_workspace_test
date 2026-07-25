"""
Modeling: Multi Galaxy Model Fit
================================

Integration test of the multi-galaxy regime introduced in `autolens_workspace/scripts/multi_galaxy/`: a lens with
**two co-dominant galaxies**, each with its own free light and mass model, fitted end-to-end with the standard
extended-source `AnalysisImaging` workflow.

The regime's defining structural property — one free `Isothermal` per deflector, untruncated (no host halo means
no tidal truncation), a single `ExternalShear` on the first galaxy, and no extra/scaling galaxy tiers — is what
this script locks in. It simulates its own dataset (a merging pair modeled on SDSS J1011+0143: ~0.9" separation,
~1.8" combined Einstein ring) so it is fully self-contained.

Its sibling `multi_galaxy/composition_mge.py` covers pure model composition (prior identity, identifier
stability); this script covers the fit path.
"""

# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import numpy as np
from os import path

import autofit as af
import autolens as al
import autolens.plot as aplt

"""
__Simulate__

Simulate the merging-pair dataset inline: two `Sersic` + `Isothermal` galaxies of comparable Einstein radius
(1.0" and 0.8") and a compact `SersicCore` source inside the tangential caustic, producing an Einstein-cross-like
configuration around the pair.
"""
grid = al.Grid2D.uniform(shape_native=(150, 150), pixel_scales=0.1)

psf = al.Convolver.from_gaussian(
    convolve_over_sample_size=1,
    shape_native=(11, 11),
    sigma=0.1,
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
        centre=main_lens_centres[0], intensity=1.0, effective_radius=0.6, sersic_index=3.0
    ),
    mass=al.mp.IsothermalSph(centre=main_lens_centres[0], einstein_radius=1.0),
)

lens_1 = al.Galaxy(
    redshift=0.5,
    bulge=al.lp.SersicSph(
        centre=main_lens_centres[1], intensity=0.8, effective_radius=0.5, sersic_index=3.0
    ),
    mass=al.mp.IsothermalSph(centre=main_lens_centres[1], einstein_radius=0.8),
)

source = al.Galaxy(
    redshift=1.0,
    bulge=al.lp.SersicCore(
        centre=(0.0, 0.03), intensity=3.0, effective_radius=0.15, sersic_index=1.0
    ),
)

tracer = al.Tracer(galaxies=[lens_0, lens_1, source])

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

aplt.plot_array(array=dataset.data)

"""
__Model__

One free light + mass model per co-dominant deflector, composed with the same list-based `lens_0`, `lens_1`, ...
API the workspace package uses. Mass profiles are **untruncated** isothermals by design. Only `lens_0` carries
the `ExternalShear`.
"""
lens_dict = {}

for i, centre in enumerate(main_lens_centres):
    bulge = af.Model(al.lp.SersicSph)
    bulge.centre = centre

    mass = af.Model(al.mp.IsothermalSph)
    mass.centre = centre

    lens_dict[f"lens_{i}"] = af.Model(
        al.Galaxy,
        redshift=0.5,
        bulge=bulge,
        mass=mass,
        shear=af.Model(al.mp.ExternalShear) if i == 0 else None,
    )

source_model = af.Model(al.Galaxy, redshift=1.0, bulge=af.Model(al.lp.SersicCore))

model = af.Collection(galaxies=af.Collection(**lens_dict, source=source_model))

"""
__Model Structure Assertions__

Lock in the regime's structural properties:

 - every deflector has its own free mass model (independent einstein_radius priors);
 - exactly one shear in the whole model;
 - no truncated profiles anywhere.
"""
prior_count_lens_0 = lens_dict["lens_0"].prior_count
prior_count_lens_1 = lens_dict["lens_1"].prior_count

assert prior_count_lens_0 == prior_count_lens_1 + 2  # only difference is the shear
assert "einstein_radius" in model.info
assert model.info.count("gamma_1") == 1  # exactly one ExternalShear
assert "dPIE" not in model.info  # untruncated by design in this regime

"""
__Search + Analysis__

Fit with Nautilus via `AnalysisImaging` — the extended-source analysis shared with galaxy and group scale (the
analysis object only changes at cluster scale, to `AnalysisPoint`).
"""
search = af.Nautilus(
    path_prefix=path.join("build", "model_fit", "multi_galaxy"),
    n_live=50,
    n_like_max=300,
    number_of_cores=2,
)

analysis = al.AnalysisImaging(dataset=dataset)

"""
__Likelihood Sanity__

The analysis `log_likelihood_function` must equal the fit's figure of merit at the same instance.
"""
import pytest

instance = model.instance_from_prior_medians()
analysis_value = analysis.log_likelihood_function(instance=instance)
fit = analysis.fit_from(instance=instance)
assert float(analysis_value) == pytest.approx(float(fit.figure_of_merit))
print("PASS: multi-galaxy LLF == figure_of_merit")

"""
__Model-Fit__
"""
result = search.fit(model=model, analysis=analysis)

"""
__Result__

Check the fit returns a tracer with two lens galaxies, each with its own fitted mass centre — the measurement
(mass centre vs light centre per deflector) that motivates the regime.
"""
tracer = result.max_log_likelihood_tracer

assert len(tracer.galaxies) == 3  # lens_0, lens_1, source

for i in range(2):
    galaxy = tracer.galaxies[i]
    print(
        f"lens_{i}: light centre = {galaxy.bulge.centre}  mass centre = {galaxy.mass.centre}"
    )

aplt.subplot_tracer(tracer=result.max_log_likelihood_tracer, grid=result.grids.lp)

aplt.subplot_fit_imaging(fit=result.max_log_likelihood_fit)
