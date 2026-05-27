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
__Sersic Family__

AbstractSersic.potential_2d_from computed via MGE decomposition.
"""

run_param_sweep("Sersic", ag.mp.Sersic, [
    dict(centre=(0.0, 0.0), ell_comps=(0.1, 0.05), intensity=0.15, effective_radius=0.8, sersic_index=1.0, mass_to_light_ratio=1.2),
    dict(centre=(0.0, 0.0), ell_comps=(0.1, 0.05), intensity=0.15, effective_radius=0.8, sersic_index=0.5, mass_to_light_ratio=1.2),
    dict(centre=(0.0, 0.0), ell_comps=(0.1, 0.05), intensity=0.15, effective_radius=0.8, sersic_index=4.0, mass_to_light_ratio=1.2),
    dict(centre=(0.0, 0.0), ell_comps=(0.3, 0.15), intensity=0.15, effective_radius=0.8, sersic_index=1.0, mass_to_light_ratio=1.2),
], grid, tol, results)

run_param_sweep("SersicSph", ag.mp.SersicSph, [
    dict(centre=(0.0, 0.0), intensity=0.15, effective_radius=0.8, sersic_index=1.0, mass_to_light_ratio=1.2),
    dict(centre=(0.0, 0.0), intensity=0.15, effective_radius=0.8, sersic_index=0.5, mass_to_light_ratio=1.2),
    dict(centre=(0.0, 0.0), intensity=0.15, effective_radius=0.8, sersic_index=4.0, mass_to_light_ratio=1.2),
], grid, tol, results)

"""
__Sersic Core__
"""

run_param_sweep("SersicCore", ag.mp.SersicCore, [
    dict(centre=(0.0, 0.0), ell_comps=(0.1, 0.05), effective_radius=0.8, sersic_index=1.0, radius_break=0.01, intensity=0.05, gamma=0.25, alpha=3.0, mass_to_light_ratio=1.2),
    dict(centre=(0.0, 0.0), ell_comps=(0.1, 0.05), effective_radius=0.8, sersic_index=0.5, radius_break=0.01, intensity=0.05, gamma=0.25, alpha=3.0, mass_to_light_ratio=1.2),
    dict(centre=(0.0, 0.0), ell_comps=(0.1, 0.05), effective_radius=0.8, sersic_index=4.0, radius_break=0.01, intensity=0.05, gamma=0.25, alpha=3.0, mass_to_light_ratio=1.2),
    dict(centre=(0.0, 0.0), ell_comps=(0.3, 0.15), effective_radius=0.8, sersic_index=1.0, radius_break=0.01, intensity=0.05, gamma=0.25, alpha=3.0, mass_to_light_ratio=1.2),
], grid, tol, results)

run_param_sweep("SersicCoreSph", ag.mp.SersicCoreSph, [
    dict(centre=(0.0, 0.0), effective_radius=0.8, sersic_index=1.0, radius_break=0.01, intensity=0.05, gamma=0.25, alpha=3.0),
    dict(centre=(0.0, 0.0), effective_radius=0.8, sersic_index=0.5, radius_break=0.01, intensity=0.05, gamma=0.25, alpha=3.0),
    dict(centre=(0.0, 0.0), effective_radius=0.8, sersic_index=4.0, radius_break=0.01, intensity=0.05, gamma=0.25, alpha=3.0),
], grid, tol, results)

"""
__Sersic Gradient__
"""

run_param_sweep("SersicGradient", ag.mp.SersicGradient, [
    dict(centre=(0.0, 0.0), ell_comps=(0.1, 0.05), intensity=0.15, effective_radius=0.8, sersic_index=1.0, mass_to_light_ratio=1.2, mass_to_light_gradient=-0.3),
    dict(centre=(0.0, 0.0), ell_comps=(0.1, 0.05), intensity=0.15, effective_radius=0.8, sersic_index=0.5, mass_to_light_ratio=1.2, mass_to_light_gradient=-0.3),
    dict(centre=(0.0, 0.0), ell_comps=(0.1, 0.05), intensity=0.15, effective_radius=0.8, sersic_index=4.0, mass_to_light_ratio=1.2, mass_to_light_gradient=-0.3),
    dict(centre=(0.0, 0.0), ell_comps=(0.3, 0.15), intensity=0.15, effective_radius=0.8, sersic_index=1.0, mass_to_light_ratio=1.2, mass_to_light_gradient=-0.3),
], grid, tol, results)

run_param_sweep("SersicGradientSph", ag.mp.SersicGradientSph, [
    dict(centre=(0.0, 0.0), intensity=0.15, effective_radius=0.8, sersic_index=1.0, mass_to_light_ratio=1.2, mass_to_light_gradient=-0.3),
    dict(centre=(0.0, 0.0), intensity=0.15, effective_radius=0.8, sersic_index=0.5, mass_to_light_ratio=1.2, mass_to_light_gradient=-0.3),
    dict(centre=(0.0, 0.0), intensity=0.15, effective_radius=0.8, sersic_index=4.0, mass_to_light_ratio=1.2, mass_to_light_gradient=-0.3),
], grid, tol, results)

"""
__Gaussian Family__

Gaussian.potential_2d_from computed via MGE decomposition.

NOTE: Only one param set is used even in full mode — Gaussian has a known
MGE limitation that makes edge-case parameter sweeps unreliable.
"""

run_param_sweep("Gaussian", ag.mp.Gaussian, [
    dict(centre=(0.0, 0.0), ell_comps=(0.1, 0.05), intensity=0.12, sigma=0.9, mass_to_light_ratio=1.1),
], grid, tol, results)

"""
__Gaussian Gradient__

NOTE: Same as Gaussian — only one param set even in full mode.
"""

run_param_sweep("GaussianGradient", ag.mp.GaussianGradient, [
    dict(centre=(0.0, 0.0), ell_comps=(0.1, 0.05), intensity=0.12, sigma=0.9, mass_to_light_ratio_base=1.1, mass_to_light_gradient=-0.3),
], grid, tol, results)

"""
__Exponential Family__

Exponential is Sersic with sersic_index=1.0.
"""

run_param_sweep("Exponential", ag.mp.Exponential, [
    dict(centre=(0.0, 0.0), ell_comps=(0.1, 0.05), intensity=0.15, effective_radius=0.8, mass_to_light_ratio=1.2),
    dict(centre=(0.0, 0.0), ell_comps=(0.1, 0.05), intensity=0.15, effective_radius=3.0, mass_to_light_ratio=1.2),
], grid, tol, results)

run_param_sweep("ExponentialSph", ag.mp.ExponentialSph, [
    dict(centre=(0.0, 0.0), intensity=0.15, effective_radius=0.8, mass_to_light_ratio=1.2),
    dict(centre=(0.0, 0.0), intensity=0.15, effective_radius=3.0, mass_to_light_ratio=1.2),
], grid, tol, results)

"""
__DevVaucouleurs Family__

DevVaucouleurs is Sersic with sersic_index=4.0.
"""

run_param_sweep("DevVaucouleurs", ag.mp.DevVaucouleurs, [
    dict(centre=(0.0, 0.0), ell_comps=(0.1, 0.05), intensity=0.15, effective_radius=0.8, mass_to_light_ratio=1.2),
    dict(centre=(0.0, 0.0), ell_comps=(0.1, 0.05), intensity=0.15, effective_radius=3.0, mass_to_light_ratio=1.2),
], grid, tol, results)

run_param_sweep("DevVaucouleursSph", ag.mp.DevVaucouleursSph, [
    dict(centre=(0.0, 0.0), intensity=0.15, effective_radius=0.8, mass_to_light_ratio=1.2),
    dict(centre=(0.0, 0.0), intensity=0.15, effective_radius=3.0, mass_to_light_ratio=1.2),
], grid, tol, results)

"""
__Chameleon Family__

Chameleon.potential_2d_from computed via MGE decomposition.
"""

run_param_sweep("Chameleon", ag.mp.Chameleon, [
    dict(centre=(0.0, 0.0), ell_comps=(0.1, 0.05), intensity=0.12, core_radius_0=0.015, core_radius_1=0.025, mass_to_light_ratio=1.1),
    dict(centre=(0.0, 0.0), ell_comps=(0.1, 0.05), intensity=0.12, core_radius_0=0.005, core_radius_1=0.2, mass_to_light_ratio=1.1),
    dict(centre=(0.0, 0.0), ell_comps=(0.3, 0.15), intensity=0.12, core_radius_0=0.015, core_radius_1=0.025, mass_to_light_ratio=1.1),
], grid, tol, results)

run_param_sweep("ChameleonSph", ag.mp.ChameleonSph, [
    dict(centre=(0.0, 0.0), intensity=0.12, core_radius_0=0.015, core_radius_1=0.025, mass_to_light_ratio=1.1),
    dict(centre=(0.0, 0.0), intensity=0.12, core_radius_0=0.005, core_radius_1=0.2, mass_to_light_ratio=1.1),
], grid, tol, results)

"""
__Summary__
"""

print("=" * 70)
print(f"Stellar Mass Profiles — Self-Consistency Results (mode={MODE})")
print("=" * 70)
print_summary_table(results)
