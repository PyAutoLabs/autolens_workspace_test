"""
Mass Profile Self-Consistency: Point Mass Profiles
===================================================

Verifies that point mass profiles (PointMass, SMBH, SMBHBinary) satisfy the
fundamental lensing relations:

    div(alpha) = 2 * kappa
    grad(psi)  = alpha
    lap(psi)   = 2 * kappa

using numerical differentiation independent of the source code.

Point masses are singular at the centre — the convergence is a delta function.
We offset the profile centre from the grid centre so that no grid pixel sits
on the singularity. The div(alpha) = 2*kappa check is only meaningful away
from the singularity (where kappa ~ 0 everywhere), so it will show as a
trivial PASS or SKIP depending on numerical noise.

PointMass has analytic potential_2d_from: psi = R_E^2 * ln(r).
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import autogalaxy as ag
from mass.util import make_grid, run_all_checks, print_summary_table, get_tolerances

"""
__Setup__

Offset the profile centre to (0.3, 0.3) so the singularity does not fall
on any grid pixel (the grid is centred at the origin).
"""

grid = make_grid()
tol = get_tolerances()
results = []

offset_centre = (0.3, 0.3)

"""
__PointMass__

potential_2d_from is analytic: psi = R_E^2 * ln(r).
"""

mp = ag.mp.PointMass(centre=offset_centre, einstein_radius=0.8)
results.append(run_all_checks("PointMass", mp, grid, tol))

"""
__SMBH__
"""

mp = ag.mp.SMBH(
    centre=offset_centre,
    mass=1.5e10,
    redshift_object=0.5,
    redshift_source=1.2,
)
results.append(run_all_checks("SMBH", mp, grid, tol))

"""
__SMBHBinary__
"""

mp = ag.mp.SMBHBinary(
    centre=offset_centre,
    separation=0.5,
    angle_binary=45.0,
    mass=1.0e10,
    mass_ratio=1.5,
    redshift_object=0.5,
    redshift_source=1.2,
)
results.append(run_all_checks("SMBHBinary", mp, grid, tol))

"""
__Summary__
"""

print("=" * 70)
print("Point Mass Profiles — Self-Consistency Results")
print("=" * 70)
print_summary_table(results)
