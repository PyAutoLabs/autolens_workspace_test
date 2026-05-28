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
__Isothermal Family__
"""

run_param_sweep(
    "Isothermal",
    ag.mp.Isothermal,
    [
        dict(centre=(0.0, 0.0), ell_comps=(0.1, 0.05), einstein_radius=1.2),
        dict(centre=(0.0, 0.0), ell_comps=(0.01, 0.005), einstein_radius=1.2),
        dict(centre=(0.0, 0.0), ell_comps=(0.3, 0.15), einstein_radius=1.2),
        dict(centre=(0.0, 0.0), ell_comps=(0.1, 0.05), einstein_radius=5.0),
        dict(centre=(0.0, 0.0), ell_comps=(0.1, 0.05), einstein_radius=0.2),
        dict(centre=(0.5, -0.3), ell_comps=(0.1, 0.05), einstein_radius=1.2),
    ],
    grid,
    tol,
    results,
)

run_param_sweep(
    "IsothermalSph",
    ag.mp.IsothermalSph,
    [
        dict(centre=(0.0, 0.0), einstein_radius=1.2),
        dict(centre=(0.0, 0.0), einstein_radius=5.0),
        dict(centre=(0.0, 0.0), einstein_radius=0.2),
    ],
    grid,
    tol,
    results,
)

run_param_sweep(
    "IsothermalCore",
    ag.mp.IsothermalCore,
    [
        dict(
            centre=(0.0, 0.0),
            ell_comps=(0.1, 0.05),
            einstein_radius=1.2,
            core_radius=0.1,
        ),
        dict(
            centre=(0.0, 0.0),
            ell_comps=(0.1, 0.05),
            einstein_radius=1.2,
            core_radius=0.001,
        ),
        dict(
            centre=(0.0, 0.0),
            ell_comps=(0.1, 0.05),
            einstein_radius=1.2,
            core_radius=0.5,
        ),
    ],
    grid,
    tol,
    results,
)

run_param_sweep(
    "IsothermalCoreSph",
    ag.mp.IsothermalCoreSph,
    [
        dict(centre=(0.0, 0.0), einstein_radius=1.2, core_radius=0.1),
    ],
    grid,
    tol,
    results,
)

"""
__Power Law Family__
"""

run_param_sweep(
    "PowerLaw",
    ag.mp.PowerLaw,
    [
        dict(centre=(0.0, 0.0), ell_comps=(0.08, 0.04), einstein_radius=1.1, slope=2.1),
        dict(centre=(0.0, 0.0), ell_comps=(0.08, 0.04), einstein_radius=1.1, slope=1.8),
        dict(centre=(0.0, 0.0), ell_comps=(0.08, 0.04), einstein_radius=1.1, slope=2.5),
        dict(centre=(0.0, 0.0), ell_comps=(0.3, 0.15), einstein_radius=1.1, slope=2.1),
    ],
    grid,
    tol,
    results,
)

run_param_sweep(
    "PowerLawSph",
    ag.mp.PowerLawSph,
    [
        dict(centre=(0.0, 0.0), einstein_radius=1.1, slope=2.1),
        dict(centre=(0.0, 0.0), einstein_radius=1.1, slope=2.0),
    ],
    grid,
    tol,
    results,
)

run_param_sweep(
    "PowerLawCore",
    ag.mp.PowerLawCore,
    [
        dict(
            centre=(0.0, 0.0),
            ell_comps=(0.08, 0.04),
            einstein_radius=1.1,
            slope=2.1,
            core_radius=0.1,
        ),
    ],
    grid,
    tol,
    results,
)

run_param_sweep(
    "PowerLawCoreSph",
    ag.mp.PowerLawCoreSph,
    [
        dict(centre=(0.0, 0.0), einstein_radius=1.1, slope=2.1, core_radius=0.1),
    ],
    grid,
    tol,
    results,
)

"""
__Power Law Broken__

potential_2d_from computed via MGE decomposition.
"""

run_param_sweep(
    "PowerLawBroken",
    ag.mp.PowerLawBroken,
    [
        dict(
            centre=(0.0, 0.0),
            ell_comps=(0.1, 0.05),
            einstein_radius=1.0,
            inner_slope=1.3,
            outer_slope=2.7,
            break_radius=0.05,
        ),
        dict(
            centre=(0.0, 0.0),
            ell_comps=(0.1, 0.05),
            einstein_radius=1.0,
            inner_slope=1.0,
            outer_slope=3.0,
            break_radius=0.1,
        ),
    ],
    grid,
    tol,
    results,
)

run_param_sweep(
    "PowerLawBrokenSph",
    ag.mp.PowerLawBrokenSph,
    [
        dict(
            centre=(0.0, 0.0),
            einstein_radius=1.0,
            inner_slope=1.3,
            outer_slope=2.7,
            break_radius=0.05,
        ),
    ],
    grid,
    tol,
    results,
)

"""
__Power Law Multipole__

convergence_2d_from returns zeros (physically correct for a pure multipole).
"""

run_param_sweep(
    "PowerLawMultipole",
    ag.mp.PowerLawMultipole,
    [
        dict(
            centre=(0.0, 0.0),
            einstein_radius=1.0,
            slope=2.0,
            m=4,
            multipole_comps=(0.01, 0.01),
        ),
        dict(
            centre=(0.0, 0.0),
            einstein_radius=1.0,
            slope=2.0,
            m=3,
            multipole_comps=(0.02, 0.01),
        ),
    ],
    grid,
    tol,
    results,
)

"""
__dPIE Family__

potential_2d_from computed via MGE decomposition.
"""

run_param_sweep(
    "dPIEMass",
    ag.mp.dPIEMass,
    [
        dict(centre=(0.0, 0.0), ell_comps=(0.1, 0.05), ra=0.02, rs=2.5, b0=0.15),
        dict(centre=(0.0, 0.0), ell_comps=(0.1, 0.05), ra=0.1, rs=5.0, b0=0.3),
    ],
    grid,
    tol,
    results,
)

run_param_sweep(
    "dPIEMassSph",
    ag.mp.dPIEMassSph,
    [
        dict(centre=(0.0, 0.0), ra=0.02, rs=2.5, b0=0.15),
    ],
    grid,
    tol,
    results,
)

run_param_sweep(
    "dPIEPotential",
    ag.mp.dPIEPotential,
    [
        dict(centre=(0.0, 0.0), ell_comps=(0.1, 0.05), ra=0.02, rs=2.5, b0=0.15),
    ],
    grid,
    tol,
    results,
)

run_param_sweep(
    "dPIEPotentialSph",
    ag.mp.dPIEPotentialSph,
    [
        dict(centre=(0.0, 0.0), ra=0.02, rs=2.5, b0=0.15),
    ],
    grid,
    tol,
    results,
)

"""
__Summary__
"""

print("=" * 70)
print(f"Total Mass Profiles — Self-Consistency Results (mode={MODE})")
print("=" * 70)
print_summary_table(results)
