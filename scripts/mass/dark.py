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
from mass.util import (
    make_grid,
    run_all_checks,
    run_param_sweep,
    print_summary_table,
    check_convergence_func_enclosed_mass,
    print_convergence_func_table,
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
__NFW Family__
"""

run_param_sweep(
    "NFW",
    ag.mp.NFW,
    [
        dict(centre=(0.0, 0.0), ell_comps=(0.08, 0.04), kappa_s=0.07, scale_radius=1.2),
        dict(centre=(0.0, 0.0), ell_comps=(0.3, 0.15), kappa_s=0.07, scale_radius=1.2),
        dict(centre=(0.0, 0.0), ell_comps=(0.08, 0.04), kappa_s=0.07, scale_radius=5.0),
        dict(centre=(0.0, 0.0), ell_comps=(0.08, 0.04), kappa_s=0.07, scale_radius=0.3),
    ],
    grid,
    tol,
    results,
)

run_param_sweep(
    "NFWSph",
    ag.mp.NFWSph,
    [
        dict(centre=(0.0, 0.0), kappa_s=0.07, scale_radius=1.2),
        dict(centre=(0.0, 0.0), kappa_s=0.2, scale_radius=1.2),
        dict(centre=(0.0, 0.0), kappa_s=0.02, scale_radius=1.2),
    ],
    grid,
    tol,
    results,
)

"""
__gNFW Family__
"""

run_param_sweep(
    "gNFW",
    ag.mp.gNFW,
    [
        dict(
            centre=(0.0, 0.0),
            ell_comps=(0.08, 0.04),
            kappa_s=0.06,
            inner_slope=1.2,
            scale_radius=1.1,
        ),
        dict(
            centre=(0.0, 0.0),
            ell_comps=(0.08, 0.04),
            kappa_s=0.06,
            inner_slope=0.5,
            scale_radius=1.1,
        ),
        dict(
            centre=(0.0, 0.0),
            ell_comps=(0.08, 0.04),
            kappa_s=0.06,
            inner_slope=2.5,
            scale_radius=1.1,
        ),
    ],
    grid,
    tol,
    results,
)

run_param_sweep(
    "gNFWSph",
    ag.mp.gNFWSph,
    [
        dict(centre=(0.0, 0.0), kappa_s=0.06, inner_slope=1.2, scale_radius=1.1),
        dict(centre=(0.0, 0.0), kappa_s=0.06, inner_slope=0.5, scale_radius=1.1),
        dict(centre=(0.0, 0.0), kappa_s=0.06, inner_slope=2.5, scale_radius=1.1),
    ],
    grid,
    tol,
    results,
)

"""
__cNFW Family__

convergence_2d_from and potential_2d_from both computed via MGE decomposition.
"""

run_param_sweep(
    "cNFW",
    ag.mp.cNFW,
    [
        dict(
            centre=(0.0, 0.0),
            ell_comps=(0.08, 0.04),
            kappa_s=0.06,
            scale_radius=1.2,
            core_radius=0.015,
        ),
        dict(
            centre=(0.0, 0.0),
            ell_comps=(0.08, 0.04),
            kappa_s=0.06,
            scale_radius=1.2,
            core_radius=0.15,
        ),
        dict(
            centre=(0.0, 0.0),
            ell_comps=(0.08, 0.04),
            kappa_s=0.06,
            scale_radius=1.2,
            core_radius=0.001,
        ),
    ],
    grid,
    tol,
    results,
)

run_param_sweep(
    "cNFWSph",
    ag.mp.cNFWSph,
    [
        dict(centre=(0.0, 0.0), kappa_s=0.06, scale_radius=1.2, core_radius=0.015),
        dict(centre=(0.0, 0.0), kappa_s=0.06, scale_radius=1.2, core_radius=0.15),
        dict(centre=(0.0, 0.0), kappa_s=0.06, scale_radius=1.2, core_radius=0.001),
    ],
    grid,
    tol,
    results,
)

"""
__NFW Truncated__

potential_2d_from computed via MGE decomposition.
"""

run_param_sweep(
    "NFWTruncatedSph",
    ag.mp.NFWTruncatedSph,
    [
        dict(centre=(0.0, 0.0), kappa_s=0.07, scale_radius=1.2, truncation_radius=3.0),
        dict(centre=(0.0, 0.0), kappa_s=0.07, scale_radius=1.2, truncation_radius=10.0),
        dict(centre=(0.0, 0.0), kappa_s=0.07, scale_radius=1.2, truncation_radius=0.5),
    ],
    grid,
    tol,
    results,
)

"""
__Summary__
"""

print("=" * 70)
print(f"Dark Matter Profiles — Self-Consistency Results (mode={MODE})")
print("=" * 70)
print_summary_table(results)

"""
__convergence_func Enclosed Mass__

Directly exercises `convergence_func` via radial mass integration (the Einstein-radius
path) — the only path that reaches cNFW's `convergence_func`, since its `convergence_2d_from`
goes through MGE-of-3D-density and never calls it. NFWSph is an analytic positive control;
cNFWSph is the profile whose `convergence_func` was newly added.
"""

cf_results = [
    check_convergence_func_enclosed_mass(
        "NFWSph",
        ag.mp.NFWSph(centre=(0.0, 0.0), kappa_s=0.2, scale_radius=1.2),
        tol,
    ),
    check_convergence_func_enclosed_mass(
        "cNFWSph",
        ag.mp.cNFWSph(
            centre=(0.0, 0.0), kappa_s=0.2, scale_radius=1.2, core_radius=0.15
        ),
        tol,
    ),
]

print_convergence_func_table(cf_results)
