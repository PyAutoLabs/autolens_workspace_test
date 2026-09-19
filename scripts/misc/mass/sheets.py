"""
Mass Profile Self-Consistency: Sheet / Perturbation Profiles
=============================================================

Verifies that sheet and external perturbation profiles (ExternalShear,
MassSheet, ExternalPotential) satisfy the fundamental lensing relations:

    div(alpha) = 2 * kappa
    grad(psi)  = alpha
    lap(psi)   = 2 * kappa

using numerical differentiation independent of the source code.

ExternalShear has physically zero convergence (pure shear field).
MassSheet has analytic potential_2d_from: psi = 0.5 * kappa_ext * r^2.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import autogalaxy as ag
from mass.util import (
    make_grid,
    run_all_checks,
    run_param_sweep,
    print_summary_table,
    get_tolerances,
    MODE,
)

"""
__Setup__
"""

grid = make_grid()
tol = get_tolerances()
results = []

"""
__External Shear__

A pure external shear field has zero convergence by definition — it only
contributes deflection angles proportional to the shear components.
The convergence SKIP is physically correct.
"""

run_param_sweep(
    "ExternalShear",
    ag.mp.ExternalShear,
    [
        dict(gamma_1=0.05, gamma_2=0.03),
        dict(gamma_1=0.2, gamma_2=0.15),
        dict(gamma_1=0.01, gamma_2=0.005),
    ],
    grid,
    tol,
    results,
)

"""
__Mass Sheet__

A uniform convergence sheet with kappa_ext. The potential is
psi = 0.5 * kappa_ext * r^2 (analytic implementation).
"""

run_param_sweep(
    "MassSheet",
    ag.mp.MassSheet,
    [
        dict(centre=(0.0, 0.0), kappa=0.1),
        dict(centre=(0.0, 0.0), kappa=0.5),
        dict(centre=(0.0, 0.0), kappa=0.01),
    ],
    grid,
    tol,
    results,
)

"""
__External Potential__

Higher-order external potential with spin-1, spin-2, and spin-3 terms.
"""

run_param_sweep(
    "ExternalPotential",
    ag.mp.ExternalPotential,
    [
        dict(
            centre=(0.0, 0.0),
            gamma_1=0.04,
            gamma_2=0.02,
            tau_1=0.01,
            tau_2=0.01,
            delta_1=0.002,
            delta_2=0.002,
        ),
        dict(
            centre=(0.0, 0.0),
            gamma_1=0.0,
            gamma_2=0.0,
            tau_1=0.03,
            tau_2=0.02,
            delta_1=0.0,
            delta_2=0.0,
        ),
        dict(
            centre=(0.0, 0.0),
            gamma_1=0.0,
            gamma_2=0.0,
            tau_1=0.0,
            tau_2=0.0,
            delta_1=0.008,
            delta_2=0.005,
        ),
        dict(
            centre=(0.0, 0.0),
            gamma_1=0.1,
            gamma_2=0.08,
            tau_1=0.05,
            tau_2=0.04,
            delta_1=0.01,
            delta_2=0.008,
        ),
    ],
    grid,
    tol,
    results,
)

"""
__MassField__

These three profiles are exactly the ones an `al.MassField` holds: the external field is a property of the
system, not of a galaxy, so it lives in its own container (a redshift plus a bag of mass profiles, carrying no
light) in the tracer's `fields=` slot rather than attached to a lens `Galaxy`.

Where the profiles are *held* cannot change the numbers — the tracer sums every deflection field over the plane.
That is asserted here against the profiles themselves: the tracer's deflections, convergence and potential equal
the direct sum of the lens galaxy's mass and the three field profiles, each evaluated on its own.

(The galaxy-attached form is still accepted by the library; its parity regression and identifier pin live in
`misc/mass/galaxy_attached_legacy.py`, which is the one script in either workspace allowed to write it.)
"""
import numpy as np

import autolens as al

field_shear = ag.mp.ExternalShear(gamma_1=0.05, gamma_2=0.03)
field_mass_sheet = ag.mp.MassSheet(centre=(0.0, 0.0), kappa=0.1)
field_potential = ag.mp.ExternalPotential(
    centre=(0.0, 0.0),
    gamma_1=0.04,
    gamma_2=0.02,
    tau_1=0.01,
    tau_2=0.01,
    delta_1=0.002,
    delta_2=0.002,
)

lens_mass = ag.mp.Isothermal(
    centre=(0.0, 0.0), ell_comps=(0.05, 0.05), einstein_radius=1.6
)

lens_no_sheet = al.Galaxy(redshift=0.5, mass=lens_mass)

src = al.Galaxy(
    redshift=1.0, bulge=al.lp.SersicSph(intensity=1.0, effective_radius=0.2)
)

field = al.MassField(
    redshift=0.5,
    shear=field_shear,
    mass_sheet=field_mass_sheet,
    potential=field_potential,
)

tracer_field = al.Tracer(galaxies=[lens_no_sheet, src], fields=[field])

field_grid = al.Grid2D.uniform(shape_native=(40, 40), pixel_scales=0.1)

profiles = [lens_mass, field_shear, field_mass_sheet, field_potential]

for quantity in ["deflections_yx_2d_from", "convergence_2d_from", "potential_2d_from"]:
    traced = np.asarray(getattr(tracer_field, quantity)(grid=field_grid))
    summed = sum(
        np.asarray(getattr(profile, quantity)(grid=field_grid)) for profile in profiles
    )

    assert np.allclose(traced, summed), quantity

    print(
        f"MassField parity  {quantity:<24} max|diff| = {np.max(np.abs(traced - summed)):.3e}"
    )

assert len(tracer_field.galaxies) == 2  # the MassField is never a galaxy
assert len(tracer_field.fields) == 1
assert tracer_field.fields[0].mass_sheet.kappa == 0.1

print("PASS: shear + sheet + potential in one MassField sum into the lens plane")

"""
__Summary__
"""

print("=" * 70)
print(f"Sheet / Perturbation Profiles — Self-Consistency Results (mode={MODE})")
print("=" * 70)
print_summary_table(results)
