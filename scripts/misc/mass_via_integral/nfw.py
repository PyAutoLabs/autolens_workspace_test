"""
NFW Mass Profile — Deflections and Potential via Integral
=========================================================

This script preserves the integral-based deflection and potential calculations
for the NFW mass profile, which were removed from the autogalaxy source code.
"""

import numpy as np
from scipy.integrate import quad
import autogalaxy as ag


"""
__NFW Deflection Function__
"""


def nfw_deflection_func(u, y, x, npow, axis_ratio, scale_radius):
    _eta_u = (1.0 / scale_radius) * np.sqrt(
        (u * ((x**2) + (y**2 / (1 - (1 - axis_ratio**2) * u))))
    )
    if _eta_u > 1:
        _eta_u_2 = (1.0 / np.sqrt(_eta_u**2 - 1)) * np.arctan(np.sqrt(_eta_u**2 - 1))
    elif _eta_u < 1:
        _eta_u_2 = (1.0 / np.sqrt(1 - _eta_u**2)) * np.arctanh(np.sqrt(1 - _eta_u**2))
    else:
        _eta_u_2 = 1
    return (
        2.0
        * (1 - _eta_u_2)
        / (_eta_u**2 - 1)
        / ((1 - (1 - axis_ratio**2) * u) ** (npow + 0.5))
    )


"""
__NFW Potential Function__
"""


def nfw_potential_func(u, y, x, axis_ratio, kappa_s, scale_radius):
    _eta_u = (1.0 / scale_radius) * np.sqrt(
        (u * ((x**2) + (y**2 / (1 - (1 - axis_ratio**2) * u))))
    )
    if _eta_u > 1:
        _eta_u_2 = (1.0 / np.sqrt(_eta_u**2 - 1)) * np.arctan(np.sqrt(_eta_u**2 - 1))
    elif _eta_u < 1:
        _eta_u_2 = (1.0 / np.sqrt(1 - _eta_u**2)) * np.arctanh(np.sqrt(1 - _eta_u**2))
    else:
        _eta_u_2 = 1
    return (
        4.0
        * kappa_s
        * scale_radius
        * (axis_ratio / 2.0)
        * (_eta_u / u)
        * ((np.log(_eta_u / 2.0) + _eta_u_2) / _eta_u)
        / ((1 - (1 - axis_ratio**2) * u) ** 0.5)
    )


"""
__Deflections via Integral__
"""


def deflections_2d_via_integral_from(mp, grid):
    transformed = mp.transformed_to_reference_frame_grid_from(grid)
    axis_ratio = mp.axis_ratio()

    def calculate_deflection_component(npow, index):
        deflection_grid = np.array(axis_ratio * transformed.array[:, index])

        for i in range(transformed.shape[0]):
            deflection_grid[i] *= (
                mp.kappa_s
                * quad(
                    nfw_deflection_func,
                    a=0.0,
                    b=1.0,
                    args=(
                        transformed.array[i, 0],
                        transformed.array[i, 1],
                        npow,
                        axis_ratio,
                        mp.scale_radius,
                    ),
                )[0]
            )

        return deflection_grid

    deflection_y = calculate_deflection_component(1.0, 0)
    deflection_x = calculate_deflection_component(0.0, 1)

    return mp.rotated_grid_from_reference_frame_from(
        np.multiply(1.0, np.vstack((deflection_y, deflection_x)).T)
    )


"""
__Potential via Integral__
"""


def potential_2d_via_integral_from(mp, grid):
    transformed = mp.transformed_to_reference_frame_grid_from(grid)
    axis_ratio = mp.axis_ratio()

    potential_grid = np.zeros(transformed.shape[0])

    for i in range(transformed.shape[0]):
        potential_grid[i] = quad(
            nfw_potential_func,
            a=0.0,
            b=1.0,
            args=(
                transformed.array[i, 0],
                transformed.array[i, 1],
                axis_ratio,
                mp.kappa_s,
                mp.scale_radius,
            ),
            epsrel=1.49e-5,
        )[0]

    return potential_grid


"""
__NFWSph Config 1__
"""

nfw = ag.mp.NFWSph(centre=(0.0, 0.0), kappa_s=1.0, scale_radius=1.0)
integral = deflections_2d_via_integral_from(
    nfw, grid=ag.Grid2DIrregular([[0.1625, 0.1625]])
)
analytic = nfw.deflections_2d_via_analytic_from(
    grid=ag.Grid2DIrregular([[0.1625, 0.1625]])
)
np.testing.assert_allclose(integral, analytic.array, rtol=1e-3)
print("NFWSph config 1: PASSED")

"""
__NFWSph Config 2__
"""

nfw = ag.mp.NFWSph(centre=(0.3, 0.2), kappa_s=2.5, scale_radius=4.0)
integral = deflections_2d_via_integral_from(
    nfw, grid=ag.Grid2DIrregular([[0.1875, 0.1625]])
)
analytic = nfw.deflections_2d_via_analytic_from(
    grid=ag.Grid2DIrregular([[0.1875, 0.1625]])
)
np.testing.assert_allclose(integral, analytic.array, rtol=1e-3)
print("NFWSph config 2: PASSED")

"""
__NFW Elliptical Config 1__
"""

nfw = ag.mp.NFW(
    centre=(0.0, 0.0),
    ell_comps=(0.0, 0.0),
    kappa_s=1.0,
    scale_radius=1.0,
)
integral = deflections_2d_via_integral_from(
    nfw, grid=ag.Grid2DIrregular([[0.1625, 0.1625]])
)
analytic = nfw.deflections_2d_via_analytic_from(
    grid=ag.Grid2DIrregular([[0.1625, 0.1625]])
)
np.testing.assert_allclose(integral, analytic.array, rtol=1e-3)
print("NFW ell config 1: PASSED")

"""
__NFW Elliptical Config 2__
"""

nfw = ag.mp.NFW(
    centre=(0.3, 0.2),
    ell_comps=(0.03669, 0.172614),
    kappa_s=2.5,
    scale_radius=4.0,
)
integral = deflections_2d_via_integral_from(
    nfw, grid=ag.Grid2DIrregular([(0.1625, 0.1625)])
)
analytic = nfw.deflections_2d_via_analytic_from(
    grid=ag.Grid2DIrregular([(0.1625, 0.1625)])
)
np.testing.assert_allclose(integral, analytic.array, rtol=1e-3)
print("NFW ell config 2: PASSED")

print("\nAll NFW integral tests passed.")
