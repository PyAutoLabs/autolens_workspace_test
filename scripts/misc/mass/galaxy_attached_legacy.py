"""
Mass: Galaxy-Attached Field (Legacy Regression)
===============================================

The external field — `ExternalShear`, `MassSheet`, `ExternalPotential` — lives in an `al.MassField` in the
tracer's / model's `fields=` slot everywhere in **PyAutoLens**' examples from 2026-09-17 onwards. The library
still accepts the older form, where those profiles are attached to a `Galaxy` as keyword arguments, and users'
own scripts written against it must keep working unchanged.

**This is the ONLY script in `autolens_workspace` or `autolens_workspace_test` allowed to attach a field to a
`Galaxy`** (the human's 2026-09-17 ruling: *"other than maybe an autolens_workspace_test integration test we
shouldn't be using a shear_galaxy or putting shears or any other field in galaxies from now on"*). Every other
`al.Galaxy(shear=...)` / `af.Model(al.Galaxy, ..., shear=...)` site in either workspace is a regression. The one
other exception is the deliberate legacy case inside `misc/interop/coolest_round_trip.py`, which exercises the
COOLEST exporter's legacy peel.

It locks in three things:

1. **Numerical parity.** A galaxy-attached tracer (`Isothermal` + `ExternalShear` + `MassSheet` on one `Galaxy`)
   and its `MassField` twin (the same `Isothermal` on the galaxy, the shear and sheet in a `MassField` at the
   same redshift) give `np.allclose` deflections, convergence, potential and traced grids. The tracer sums every
   deflection field over the plane, so where the profiles are *held* cannot change the numbers.

2. **Identifier stability.** The galaxy-attached *model* still hashes to the identifier frozen below. A user with
   an `output/` folder produced by a galaxy-attached model must keep resuming it after the `MassField` work.

3. **The fit path.** Both models fit through `al.AnalysisImaging` and return a result.
"""

# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

from os import path

import numpy as np

import autofit as af
import autolens as al

"""
__Profiles__

The three mass components, built once and shared by both constructions so the parity check compares the
*containers*, never two different sets of numbers.
"""
mass = al.mp.Isothermal(
    centre=(0.0, 0.0),
    ell_comps=(0.05, 0.05),
    einstein_radius=1.6,
)
shear = al.mp.ExternalShear(gamma_1=0.05, gamma_2=-0.03)
mass_sheet = al.mp.MassSheet(centre=(0.0, 0.0), kappa=0.05)

source = al.Galaxy(
    redshift=1.0,
    bulge=al.lp.SersicCore(
        centre=(0.0, 0.0),
        ell_comps=(0.0, 0.1),
        intensity=4.0,
        effective_radius=0.1,
        sersic_index=1.0,
    ),
)

"""
__Tracers: Galaxy-Attached vs MassField__

The legacy form attaches `shear` and `mass_sheet` to the lens `Galaxy` as keyword arguments. The current form
puts the lens' own mass on the `Galaxy` and hands the external field to a separate `al.MassField` at the same
redshift, passed to the tracer's `fields=` argument.
"""
lens_attached = al.Galaxy(
    redshift=0.5,
    mass=mass,
    shear=shear,
    mass_sheet=mass_sheet,
)

tracer_attached = al.Tracer(galaxies=[lens_attached, source])

lens_field = al.Galaxy(redshift=0.5, mass=mass)

field = al.MassField(redshift=0.5, shear=shear, mass_sheet=mass_sheet)

tracer_field = al.Tracer(galaxies=[lens_field, source], fields=[field])

"""
The `MassField` is never a member of `tracer.galaxies` — it is only merged into the plane at its redshift.
"""
assert len(tracer_attached.galaxies) == 2
assert len(tracer_field.galaxies) == 2
assert len(tracer_field.fields) == 1
assert tracer_field.fields[0].shear.gamma_1 == shear.gamma_1

"""
__Parity__

Deflections, convergence, potential and the traced grids must agree to machine precision.
"""
grid = al.Grid2D.uniform(shape_native=(50, 50), pixel_scales=0.1)

deflections_attached = np.asarray(tracer_attached.deflections_yx_2d_from(grid=grid))
deflections_field = np.asarray(tracer_field.deflections_yx_2d_from(grid=grid))

convergence_attached = np.asarray(tracer_attached.convergence_2d_from(grid=grid))
convergence_field = np.asarray(tracer_field.convergence_2d_from(grid=grid))

potential_attached = np.asarray(tracer_attached.potential_2d_from(grid=grid))
potential_field = np.asarray(tracer_field.potential_2d_from(grid=grid))

traced_attached = tracer_attached.traced_grid_2d_list_from(grid=grid)
traced_field = tracer_field.traced_grid_2d_list_from(grid=grid)

assert np.allclose(deflections_attached, deflections_field)
assert np.allclose(convergence_attached, convergence_field)
assert np.allclose(potential_attached, potential_field)

assert len(traced_attached) == len(traced_field)

traced_max_diff = 0.0

for plane_attached, plane_field in zip(traced_attached, traced_field):
    plane_attached = np.asarray(plane_attached)
    plane_field = np.asarray(plane_field)

    assert np.allclose(plane_attached, plane_field)

    traced_max_diff = max(
        traced_max_diff, float(np.max(np.abs(plane_attached - plane_field)))
    )

print(
    f"PASS: parity  deflections {np.max(np.abs(deflections_attached - deflections_field)):.3e}  "
    f"convergence {np.max(np.abs(convergence_attached - convergence_field)):.3e}  "
    f"potential {np.max(np.abs(potential_attached - potential_field)):.3e}  "
    f"traced grids {traced_max_diff:.3e}"
)

"""
__Identifier Pin__

The galaxy-attached model's **PyAutoFit** identifier. A model identifier is a function of the model *and* the
active **PyAutoFit** configuration, so this constant is the value under `autolens_workspace_test/config/` — this
script must be run from the workspace root, as every script here is.

Computed on library `main` (PyAutoGalaxy `33714b80` / PyAutoLens `71973806`), i.e. with the `MassField` work of
phases 1 and 2 already in. It is a workspace-side backwards-compatibility guard: the galaxy-attached form must
keep hashing to the same value, because that hash is the name of the `output/` folder a user's existing results
live in. If this assertion fails, `Galaxy` (or a prior default) changed in a way **PyAutoFit** sees and the
change must be undone — never update the constant to match a new value.

The library-side twin of this pin lives in
`PyAutoLens/test_autolens/lens/test_tracer_fields.py::test__regression__the_model_identifier_is_unchanged_by_the_fields_slot`,
which pins the same model under the *library's* shipped configuration (`fef2697b5c32ba56bb18a7baecb7b0f6`).
"""
GALAXY_ATTACHED_IDENTIFIER = "35ebe9353118bcc0c7b2d577ce2639ee"

model_attached = af.Collection(
    galaxies=af.Collection(
        lens=af.Model(
            al.Galaxy,
            redshift=0.5,
            mass=af.Model(al.mp.Isothermal),
            shear=af.Model(al.mp.ExternalShear),
        ),
        source=af.Model(al.Galaxy, redshift=1.0, bulge=af.Model(al.lp.Sersic)),
    )
)

assert model_attached.identifier == GALAXY_ATTACHED_IDENTIFIER, (
    f"The galaxy-attached model identifier moved from {GALAXY_ATTACHED_IDENTIFIER} to "
    f"{model_attached.identifier}. Every existing user `output/` folder produced by a galaxy-attached "
    f"model is orphaned by this. Undo the change that caused it; do not update the constant."
)

print(f"PASS: galaxy-attached model identifier pinned at {GALAXY_ATTACHED_IDENTIFIER}")

"""
The `MassField` model composes a *different* model and therefore has its own identifier — it is not, and must
not be, the same value.

It is written in the older `fields=af.Collection(field=...)` form on purpose. Every other model in
`autolens_workspace` and `autolens_workspace_test` now sits in the flat `fields=field` slot, so this is the
workspaces' last regression witness that the library still accepts a collection there. Do not migrate it.
"""
model_field = af.Collection(
    galaxies=af.Collection(
        lens=af.Model(al.Galaxy, redshift=0.5, mass=af.Model(al.mp.Isothermal)),
        source=af.Model(al.Galaxy, redshift=1.0, bulge=af.Model(al.lp.Sersic)),
    ),
    fields=af.Collection(
        field=af.Model(al.MassField, redshift=0.5, shear=af.Model(al.mp.ExternalShear))
    ),
)

assert model_field.identifier != GALAXY_ATTACHED_IDENTIFIER
assert model_attached.prior_count == model_field.prior_count

"""
__Dataset__

A small dataset simulated in-script from the `MassField` tracer above, so the script is self-contained and never
depends on a committed dataset.
"""
simulator_grid = al.Grid2D.uniform(shape_native=(60, 60), pixel_scales=0.2)

psf = al.Convolver.from_gaussian(
    shape_native=(11, 11), sigma=0.2, pixel_scales=simulator_grid.pixel_scales
)

simulator = al.SimulatorImaging(
    exposure_time=300.0,
    psf=psf,
    background_sky_level=0.1,
    add_poisson_noise_to_data=True,
    noise_seed=1,
)

dataset = simulator.via_tracer_from(tracer=tracer_field, grid=simulator_grid)

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=3.0,
)

dataset = dataset.apply_mask(mask=mask)

"""
__Model-Fit__

Both models fit through the same `al.AnalysisImaging`, proving the galaxy-attached form is still a fittable
model and not merely a constructible object.
"""
analysis = al.AnalysisImaging(dataset=dataset)

for name, model in [("galaxy_attached", model_attached), ("mass_field", model_field)]:
    search = af.Nautilus(
        path_prefix=path.join("build", "misc", "mass", "galaxy_attached_legacy"),
        name=name,
        n_live=50,
        n_like_max=100,
        number_of_cores=1,
    )

    result = search.fit(model=model, analysis=analysis)

    assert result is not None, f"{name}: search.fit returned no result"

    print(f"PASS: {name} fit returned a result")

print("All galaxy-attached legacy regression checks passed.")
