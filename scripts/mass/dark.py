"""
Mass Profile Self-Consistency: Dark Matter Profiles
====================================================

Verifies that dark matter mass profiles (NFW, gNFW, cNFW families)
satisfy the fundamental lensing relations:

    div(alpha) = 2 * kappa
    grad(psi)  = alpha
    lap(psi)   = 2 * kappa

using numerical differentiation independent of the source code.

MCR, Scatter, and Virial variants are skipped — they derive parameters from
mass-concentration relations but delegate to the same underlying NFW/gNFW/cNFW
math for deflections, convergence, and potential.
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
__NFW Family__
"""

mp = ag.mp.NFW(
    centre=(0.0, 0.0),
    ell_comps=(0.08, 0.04),
    kappa_s=0.07,
    scale_radius=1.2,
)
results.append(run_all_checks("NFW", mp, grid, tol))

mp = ag.mp.NFWSph(centre=(0.0, 0.0), kappa_s=0.07, scale_radius=1.2)
results.append(run_all_checks("NFWSph", mp, grid, tol))

"""
__gNFW Family__
"""

mp = ag.mp.gNFW(
    centre=(0.0, 0.0),
    ell_comps=(0.08, 0.04),
    kappa_s=0.06,
    inner_slope=1.2,
    scale_radius=1.1,
)
results.append(run_all_checks("gNFW", mp, grid, tol))

mp = ag.mp.gNFWSph(
    centre=(0.0, 0.0), kappa_s=0.06, inner_slope=1.2, scale_radius=1.1
)
results.append(run_all_checks("gNFWSph", mp, grid, tol))

"""
__cNFW Family__

convergence_2d_from and potential_2d_from both return zeros.
"""

mp = ag.mp.cNFW(
    centre=(0.0, 0.0),
    ell_comps=(0.08, 0.04),
    kappa_s=0.06,
    scale_radius=1.2,
    core_radius=0.015,
)
results.append(run_all_checks("cNFW", mp, grid, tol))

mp = ag.mp.cNFWSph(
    centre=(0.0, 0.0), kappa_s=0.06, scale_radius=1.2, core_radius=0.015
)
results.append(run_all_checks("cNFWSph", mp, grid, tol))

"""
__NFW Truncated__

potential_2d_from returns zeros.
"""

mp = ag.mp.NFWTruncatedSph(
    centre=(0.0, 0.0),
    kappa_s=0.07,
    scale_radius=1.2,
    truncation_radius=3.0,
)
results.append(run_all_checks("NFWTruncatedSph", mp, grid, tol))

"""
__Summary__
"""

print("=" * 70)
print("Dark Matter Profiles — Self-Consistency Results")
print("=" * 70)
print_summary_table(results)
