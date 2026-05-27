"""
Mass Profile Self-Consistency: Stellar Profiles
================================================

Verifies that stellar mass profiles (Sersic, Gaussian, Chameleon families)
satisfy the fundamental lensing relations:

    div(alpha) = 2 * kappa
    grad(psi)  = alpha
    lap(psi)   = 2 * kappa

using numerical differentiation independent of the source code.

Stellar profiles use ``mass_to_light_ratio`` and ``intensity`` instead of
``einstein_radius``.
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
__Sersic Family__

AbstractSersic.potential_2d_from computed via MGE decomposition.
"""

mp = ag.mp.Sersic(
    centre=(0.0, 0.0),
    ell_comps=(0.1, 0.05),
    intensity=0.15,
    effective_radius=0.8,
    sersic_index=1.0,
    mass_to_light_ratio=1.2,
)
results.append(run_all_checks("Sersic", mp, grid, tol))

mp = ag.mp.SersicSph(
    centre=(0.0, 0.0),
    intensity=0.15,
    effective_radius=0.8,
    sersic_index=1.0,
    mass_to_light_ratio=1.2,
)
results.append(run_all_checks("SersicSph", mp, grid, tol))

"""
__Sersic Core__
"""

mp = ag.mp.SersicCore(
    centre=(0.0, 0.0),
    ell_comps=(0.1, 0.05),
    effective_radius=0.8,
    sersic_index=1.0,
    radius_break=0.01,
    intensity=0.05,
    gamma=0.25,
    alpha=3.0,
    mass_to_light_ratio=1.2,
)
results.append(run_all_checks("SersicCore", mp, grid, tol))

mp = ag.mp.SersicCoreSph(
    centre=(0.0, 0.0),
    effective_radius=0.8,
    sersic_index=1.0,
    radius_break=0.01,
    intensity=0.05,
    gamma=0.25,
    alpha=3.0,
)
results.append(run_all_checks("SersicCoreSph", mp, grid, tol))

"""
__Sersic Gradient__
"""

mp = ag.mp.SersicGradient(
    centre=(0.0, 0.0),
    ell_comps=(0.1, 0.05),
    intensity=0.15,
    effective_radius=0.8,
    sersic_index=1.0,
    mass_to_light_ratio=1.2,
    mass_to_light_gradient=-0.3,
)
results.append(run_all_checks("SersicGradient", mp, grid, tol))

mp = ag.mp.SersicGradientSph(
    centre=(0.0, 0.0),
    intensity=0.15,
    effective_radius=0.8,
    sersic_index=1.0,
    mass_to_light_ratio=1.2,
    mass_to_light_gradient=-0.3,
)
results.append(run_all_checks("SersicGradientSph", mp, grid, tol))

"""
__Gaussian Family__

Gaussian.potential_2d_from computed via MGE decomposition.
"""

mp = ag.mp.Gaussian(
    centre=(0.0, 0.0),
    ell_comps=(0.1, 0.05),
    intensity=0.12,
    sigma=0.9,
    mass_to_light_ratio=1.1,
)
results.append(run_all_checks("Gaussian", mp, grid, tol))

"""
__Gaussian Gradient__
"""

mp = ag.mp.GaussianGradient(
    centre=(0.0, 0.0),
    ell_comps=(0.1, 0.05),
    intensity=0.12,
    sigma=0.9,
    mass_to_light_ratio_base=1.1,
    mass_to_light_gradient=-0.3,
)
results.append(run_all_checks("GaussianGradient", mp, grid, tol))

"""
__Exponential Family__

Exponential is Sersic with sersic_index=1.0.
"""

mp = ag.mp.Exponential(
    centre=(0.0, 0.0),
    ell_comps=(0.1, 0.05),
    intensity=0.15,
    effective_radius=0.8,
    mass_to_light_ratio=1.2,
)
results.append(run_all_checks("Exponential", mp, grid, tol))

mp = ag.mp.ExponentialSph(
    centre=(0.0, 0.0),
    intensity=0.15,
    effective_radius=0.8,
    mass_to_light_ratio=1.2,
)
results.append(run_all_checks("ExponentialSph", mp, grid, tol))

"""
__DevVaucouleurs Family__

DevVaucouleurs is Sersic with sersic_index=4.0.
"""

mp = ag.mp.DevVaucouleurs(
    centre=(0.0, 0.0),
    ell_comps=(0.1, 0.05),
    intensity=0.15,
    effective_radius=0.8,
    mass_to_light_ratio=1.2,
)
results.append(run_all_checks("DevVaucouleurs", mp, grid, tol))

mp = ag.mp.DevVaucouleursSph(
    centre=(0.0, 0.0),
    intensity=0.15,
    effective_radius=0.8,
    mass_to_light_ratio=1.2,
)
results.append(run_all_checks("DevVaucouleursSph", mp, grid, tol))

"""
__Chameleon Family__

Chameleon.potential_2d_from computed via MGE decomposition.
"""

mp = ag.mp.Chameleon(
    centre=(0.0, 0.0),
    ell_comps=(0.1, 0.05),
    intensity=0.12,
    core_radius_0=0.015,
    core_radius_1=0.025,
    mass_to_light_ratio=1.1,
)
results.append(run_all_checks("Chameleon", mp, grid, tol))

mp = ag.mp.ChameleonSph(
    centre=(0.0, 0.0),
    intensity=0.12,
    core_radius_0=0.015,
    core_radius_1=0.025,
    mass_to_light_ratio=1.1,
)
results.append(run_all_checks("ChameleonSph", mp, grid, tol))

"""
__Summary__
"""

print("=" * 70)
print("Stellar Mass Profiles — Self-Consistency Results")
print("=" * 70)
print_summary_table(results)
