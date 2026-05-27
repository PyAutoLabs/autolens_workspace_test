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
    make_grid, run_all_checks, run_param_sweep, print_summary_table,
    get_tolerances, MODE,
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

run_param_sweep("ExternalShear", ag.mp.ExternalShear, [
    dict(gamma_1=0.05, gamma_2=0.03),
    dict(gamma_1=0.2, gamma_2=0.15),
    dict(gamma_1=0.01, gamma_2=0.005),
], grid, tol, results)

"""
__Mass Sheet__

A uniform convergence sheet with kappa_ext. The potential is
psi = 0.5 * kappa_ext * r^2 (analytic implementation).
"""

run_param_sweep("MassSheet", ag.mp.MassSheet, [
    dict(centre=(0.0, 0.0), kappa=0.1),
    dict(centre=(0.0, 0.0), kappa=0.5),
    dict(centre=(0.0, 0.0), kappa=0.01),
], grid, tol, results)

"""
__External Potential__

Higher-order external potential with spin-1, spin-2, and spin-3 terms.
"""

run_param_sweep("ExternalPotential", ag.mp.ExternalPotential, [
    dict(centre=(0.0, 0.0), gamma_1=0.04, gamma_2=0.02, tau_1=0.01, tau_2=0.01, delta_1=0.002, delta_2=0.002),
    dict(centre=(0.0, 0.0), gamma_1=0.0, gamma_2=0.0, tau_1=0.03, tau_2=0.02, delta_1=0.0, delta_2=0.0),
    dict(centre=(0.0, 0.0), gamma_1=0.0, gamma_2=0.0, tau_1=0.0, tau_2=0.0, delta_1=0.008, delta_2=0.005),
    dict(centre=(0.0, 0.0), gamma_1=0.1, gamma_2=0.08, tau_1=0.05, tau_2=0.04, delta_1=0.01, delta_2=0.008),
], grid, tol, results)

"""
__Summary__
"""

print("=" * 70)
print(f"Sheet / Perturbation Profiles — Self-Consistency Results (mode={MODE})")
print("=" * 70)
print_summary_table(results)
