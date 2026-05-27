"""
Mass Profile Self-Consistency: Total Profiles
==============================================

Verifies that total mass profiles (Isothermal, PowerLaw, dPIE families)
satisfy the fundamental lensing relations:

    div(alpha) = 2 * kappa
    grad(psi)  = alpha
    lap(psi)   = 2 * kappa

using numerical differentiation independent of the source code.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import autogalaxy as ag
from mass.util import make_grid, run_all_checks, print_summary_table, get_tolerances

"""
__Setup__
"""

grid = make_grid()
tol = get_tolerances()
results = []

"""
__Isothermal Family__
"""

mp = ag.mp.Isothermal(
    centre=(0.0, 0.0), ell_comps=(0.1, 0.05), einstein_radius=1.2
)
results.append(run_all_checks("Isothermal", mp, grid, tol))

mp = ag.mp.IsothermalSph(centre=(0.0, 0.0), einstein_radius=1.2)
results.append(run_all_checks("IsothermalSph", mp, grid, tol))

mp = ag.mp.IsothermalCore(
    centre=(0.0, 0.0),
    ell_comps=(0.1, 0.05),
    einstein_radius=1.2,
    core_radius=0.1,
)
results.append(run_all_checks("IsothermalCore", mp, grid, tol))

mp = ag.mp.IsothermalCoreSph(
    centre=(0.0, 0.0), einstein_radius=1.2, core_radius=0.1
)
results.append(run_all_checks("IsothermalCoreSph", mp, grid, tol))

"""
__Power Law Family__
"""

mp = ag.mp.PowerLaw(
    centre=(0.0, 0.0),
    ell_comps=(0.08, 0.04),
    einstein_radius=1.1,
    slope=2.1,
)
results.append(run_all_checks("PowerLaw", mp, grid, tol))

mp = ag.mp.PowerLawSph(centre=(0.0, 0.0), einstein_radius=1.1, slope=2.1)
results.append(run_all_checks("PowerLawSph", mp, grid, tol))

mp = ag.mp.PowerLawCore(
    centre=(0.0, 0.0),
    ell_comps=(0.08, 0.04),
    einstein_radius=1.1,
    slope=2.1,
    core_radius=0.1,
)
results.append(run_all_checks("PowerLawCore", mp, grid, tol))

mp = ag.mp.PowerLawCoreSph(
    centre=(0.0, 0.0), einstein_radius=1.1, slope=2.1, core_radius=0.1
)
results.append(run_all_checks("PowerLawCoreSph", mp, grid, tol))

"""
__Power Law Broken__

potential_2d_from computed via MGE decomposition.
"""

mp = ag.mp.PowerLawBroken(
    centre=(0.0, 0.0),
    ell_comps=(0.1, 0.05),
    einstein_radius=1.0,
    inner_slope=1.3,
    outer_slope=2.7,
    break_radius=0.05,
)
results.append(run_all_checks("PowerLawBroken", mp, grid, tol))

mp = ag.mp.PowerLawBrokenSph(
    centre=(0.0, 0.0),
    einstein_radius=1.0,
    inner_slope=1.3,
    outer_slope=2.7,
    break_radius=0.05,
)
results.append(run_all_checks("PowerLawBrokenSph", mp, grid, tol))

"""
__Power Law Multipole__

convergence_2d_from returns zeros (physically correct for a pure multipole
perturbation). div(alpha) and lap(psi) checks will SKIP.
"""

mp = ag.mp.PowerLawMultipole(
    centre=(0.0, 0.0),
    einstein_radius=1.0,
    slope=2.0,
    m=4,
    multipole_comps=(0.01, 0.01),
)
results.append(run_all_checks("PowerLawMultipole", mp, grid, tol))

"""
__dPIE Family__

dPIEMass and dPIEPotential potential_2d_from computed via MGE decomposition.
"""

mp = ag.mp.dPIEMass(
    centre=(0.0, 0.0),
    ell_comps=(0.1, 0.05),
    ra=0.02,
    rs=2.5,
    b0=0.15,
)
results.append(run_all_checks("dPIEMass", mp, grid, tol))

mp = ag.mp.dPIEMassSph(
    centre=(0.0, 0.0),
    ra=0.02,
    rs=2.5,
    b0=0.15,
)
results.append(run_all_checks("dPIEMassSph", mp, grid, tol))

mp = ag.mp.dPIEPotential(
    centre=(0.0, 0.0),
    ell_comps=(0.1, 0.05),
    ra=0.02,
    rs=2.5,
    b0=0.15,
)
results.append(run_all_checks("dPIEPotential", mp, grid, tol))

mp = ag.mp.dPIEPotentialSph(
    centre=(0.0, 0.0),
    ra=0.02,
    rs=2.5,
    b0=0.15,
)
results.append(run_all_checks("dPIEPotentialSph", mp, grid, tol))

"""
__Summary__
"""

print("=" * 70)
print("Total Mass Profiles — Self-Consistency Results")
print("=" * 70)
print_summary_table(results)
