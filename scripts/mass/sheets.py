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
MassSheet has zero potential_2d_from in the source code (placeholder).
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
__External Shear__

A pure external shear field has zero convergence by definition — it only
contributes deflection angles proportional to the shear components.
The convergence SKIP is physically correct.
"""

mp = ag.mp.ExternalShear(gamma_1=0.05, gamma_2=0.03)
results.append(run_all_checks("ExternalShear", mp, grid, tol))

"""
__Mass Sheet__

A uniform convergence sheet with kappa_ext. The potential is
psi = 0.5 * kappa_ext * r^2, but potential_2d_from currently returns zeros.
"""

mp = ag.mp.MassSheet(centre=(0.0, 0.0), kappa=0.1)
results.append(run_all_checks("MassSheet", mp, grid, tol))

"""
__External Potential__

Higher-order external potential with spin-1, spin-2, and spin-3 terms.
"""

mp = ag.mp.ExternalPotential(
    centre=(0.0, 0.0),
    gamma_1=0.04,
    gamma_2=0.02,
    tau_1=0.01,
    tau_2=0.01,
    delta_1=0.002,
    delta_2=0.002,
)
results.append(run_all_checks("ExternalPotential", mp, grid, tol))

"""
__Summary__
"""

print("=" * 70)
print("Sheet / Perturbation Profiles — Self-Consistency Results")
print("=" * 70)
print_summary_table(results)
