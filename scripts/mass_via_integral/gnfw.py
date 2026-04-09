"""
gNFW Mass Profile — Deflections, Potential, and Convergence via Integral
========================================================================

This script preserves the integral-based calculations for the gNFW mass profile,
which were removed from the autogalaxy source code.
"""
import numpy as np
from scipy import special
from scipy.integrate import quad
import autogalaxy as ag


epsrel = 1.49e-6


"""
__Tabulate Integral__

Helper to set up the tabulation grid for the gNFW integral.
"""


def tabulate_integral(mp, grid, tabulate_bins):
    eta_min = 1.0e-4
    eta_max = 1.05 * np.max(mp.elliptical_radii_grid_from(grid=grid))

    minimum_log_eta = np.log10(eta_min)
    maximum_log_eta = np.log10(eta_max)
    bin_size = (maximum_log_eta - minimum_log_eta) / (tabulate_bins - 1)

    return eta_min, eta_max, minimum_log_eta, maximum_log_eta, bin_size


"""
__Surface Density Integrand (Elliptical gNFW)__
"""


def surface_density_integrand(x, kappa_radius, scale_radius, inner_slope):
    return (
        (3 - inner_slope)
        * (x + kappa_radius / scale_radius) ** (inner_slope - 4)
        * (1 - np.sqrt(1 - x * x))
    )


"""
__gNFW Deflection Function (Elliptical, tabulated)__
"""


def gnfw_deflection_func(
    u,
    y,
    x,
    npow,
    axis_ratio,
    minimum_log_eta,
    maximum_log_eta,
    tabulate_bins,
    surface_density_integral,
):
    _eta_u = np.sqrt((u * ((x**2) + (y**2 / (1 - (1 - axis_ratio**2) * u)))))
    bin_size = (maximum_log_eta - minimum_log_eta) / (tabulate_bins - 1)
    i = 1 + int((np.log10(_eta_u) - minimum_log_eta) / bin_size)
    r1 = 10.0 ** (minimum_log_eta + (i - 1) * bin_size)
    r2 = r1 * 10.0**bin_size
    kap = surface_density_integral[i] + (
        surface_density_integral[i + 1] - surface_density_integral[i]
    ) * (_eta_u - r1) / (r2 - r1)
    return kap / (1.0 - (1.0 - axis_ratio**2) * u) ** (npow + 0.5)


"""
__gNFW Potential Function (Elliptical, tabulated)__
"""


def deflection_integrand_for_potential(x, kappa_radius, scale_radius, inner_slope):
    return (x + kappa_radius / scale_radius) ** (inner_slope - 3) * (
        (1 - np.sqrt(1 - x**2)) / x
    )


def gnfw_potential_func(
    u,
    y,
    x,
    axis_ratio,
    minimum_log_eta,
    maximum_log_eta,
    tabulate_bins,
    potential_integral,
):
    _eta_u = np.sqrt((u * ((x**2) + (y**2 / (1 - (1 - axis_ratio**2) * u)))))
    bin_size = (maximum_log_eta - minimum_log_eta) / (tabulate_bins - 1)
    i = 1 + int((np.log10(_eta_u) - minimum_log_eta) / bin_size)
    r1 = 10.0 ** (minimum_log_eta + (i - 1) * bin_size)
    r2 = r1 * 10.0**bin_size
    angle = potential_integral[i] + (
        potential_integral[i + 1] - potential_integral[i]
    ) * (_eta_u - r1) / (r2 - r1)
    return _eta_u * (angle / u) / (1.0 - (1.0 - axis_ratio**2) * u) ** 0.5


"""
__gNFWSph Deflection Integrand and Function__
"""


def deflection_integrand_sph(y, eta, inner_slope):
    return (y + eta) ** (inner_slope - 3) * ((1 - np.sqrt(1 - y**2)) / y)


def deflection_func_sph(mp, eta):
    integral_y_2 = quad(
        deflection_integrand_sph,
        a=0.0,
        b=1.0,
        args=(eta, mp.inner_slope),
        epsrel=1.49e-6,
    )[0]
    return eta ** (2 - mp.inner_slope) * (
        (1.0 / (3 - mp.inner_slope))
        * special.hyp2f1(
            3 - mp.inner_slope, 3 - mp.inner_slope, 4 - mp.inner_slope, -eta
        )
        + integral_y_2
    )


"""
__Deflections via Integral (Elliptical gNFW)__
"""


def deflections_2d_via_integral_from(mp, grid, tabulate_bins=1000):
    transformed = mp.transformed_to_reference_frame_grid_from(grid)
    axis_ratio = mp.axis_ratio()

    (
        eta_min,
        eta_max,
        minimum_log_eta,
        maximum_log_eta,
        bin_size,
    ) = tabulate_integral(mp, transformed, tabulate_bins)

    surface_density_integral_arr = np.zeros((tabulate_bins,))

    for i in range(tabulate_bins):
        eta = 10.0 ** (minimum_log_eta + (i - 1) * bin_size)

        integral = quad(
            surface_density_integrand,
            a=0.0,
            b=1.0,
            args=(eta, mp.scale_radius, mp.inner_slope),
            epsrel=epsrel,
        )[0]

        surface_density_integral_arr[i] = (
            (eta / mp.scale_radius) ** (1 - mp.inner_slope)
        ) * (((1 + eta / mp.scale_radius) ** (mp.inner_slope - 3)) + integral)

    def calculate_deflection_component(npow, yx_index):
        deflection_grid = np.zeros(transformed.shape[0])

        for i in range(transformed.shape[0]):
            deflection_grid[i] = (
                2.0
                * mp.kappa_s
                * axis_ratio
                * transformed.array[i, yx_index]
                * quad(
                    gnfw_deflection_func,
                    a=0.0,
                    b=1.0,
                    args=(
                        transformed.array[i, 0],
                        transformed.array[i, 1],
                        npow,
                        axis_ratio,
                        minimum_log_eta,
                        maximum_log_eta,
                        tabulate_bins,
                        surface_density_integral_arr,
                    ),
                    epsrel=epsrel,
                )[0]
            )

        return deflection_grid

    deflection_y = calculate_deflection_component(npow=1.0, yx_index=0)
    deflection_x = calculate_deflection_component(npow=0.0, yx_index=1)

    return mp.rotated_grid_from_reference_frame_from(
        np.multiply(1.0, np.vstack((deflection_y, deflection_x)).T),
    )


"""
__Deflections via Integral (Spherical gNFWSph)__
"""


def deflections_2d_via_integral_sph_from(mp, grid):
    transformed = mp.transformed_to_reference_frame_grid_from(grid)

    eta = np.multiply(
        1.0 / mp.scale_radius,
        mp.radial_grid_from(transformed, is_transformed=True).array,
    )

    deflection_grid = np.zeros(transformed.shape[0])

    for i in range(transformed.shape[0]):
        deflection_grid[i] = np.multiply(
            4.0 * mp.kappa_s * mp.scale_radius, deflection_func_sph(mp, eta[i])
        )

    return mp._cartesian_grid_via_radial_from(
        grid=transformed, radius=deflection_grid, xp=np
    )


"""
__Potential via Integral (Elliptical gNFW)__
"""


def potential_2d_via_integral_from(mp, grid, tabulate_bins=1000):
    transformed = mp.transformed_to_reference_frame_grid_from(grid)
    axis_ratio = mp.axis_ratio()

    (
        eta_min,
        eta_max,
        minimum_log_eta,
        maximum_log_eta,
        bin_size,
    ) = tabulate_integral(mp, transformed, tabulate_bins)

    deflection_integral = np.zeros((tabulate_bins,))

    for i in range(tabulate_bins):
        eta = 10.0 ** (minimum_log_eta + (i - 1) * bin_size)

        integral = quad(
            deflection_integrand_for_potential,
            a=0.0,
            b=1.0,
            args=(eta, mp.scale_radius, mp.inner_slope),
            epsrel=epsrel,
        )[0]

        deflection_integral[i] = (
            (eta / mp.scale_radius) ** (2 - mp.inner_slope)
        ) * (
            (1.0 / (3 - mp.inner_slope))
            * special.hyp2f1(
                3 - mp.inner_slope,
                3 - mp.inner_slope,
                4 - mp.inner_slope,
                -(eta / mp.scale_radius),
            )
            + integral
        )

    potential_grid = np.zeros(transformed.shape[0])

    for i in range(transformed.shape[0]):
        potential_grid[i] = (2.0 * mp.kappa_s * axis_ratio) * quad(
            gnfw_potential_func,
            a=0.0,
            b=1.0,
            args=(
                transformed.array[i, 0],
                transformed.array[i, 1],
                axis_ratio,
                minimum_log_eta,
                maximum_log_eta,
                tabulate_bins,
                deflection_integral,
            ),
            epsrel=epsrel,
        )[0]

    return potential_grid


"""
__gNFWSph Config 1__
"""

mp = ag.mp.gNFWSph(
    centre=(0.0, 0.0), kappa_s=1.0, inner_slope=0.5, scale_radius=8.0
)
integral = deflections_2d_via_integral_sph_from(
    mp, grid=ag.Grid2DIrregular([[0.1875, 0.1625]])
)
mge = mp.deflections_2d_via_mge_from(grid=ag.Grid2DIrregular([[0.1875, 0.1625]]))
np.testing.assert_allclose(integral, mge.array, rtol=1e-3)
print("gNFWSph config 1: PASSED")

"""
__gNFWSph Config 2__
"""

mp = ag.mp.gNFWSph(
    centre=(0.3, 0.2), kappa_s=2.5, inner_slope=1.5, scale_radius=4.0
)
integral = deflections_2d_via_integral_sph_from(
    mp, grid=ag.Grid2DIrregular([[0.1875, 0.1625]])
)
mge = mp.deflections_2d_via_mge_from(grid=ag.Grid2DIrregular([[0.1875, 0.1625]]))
np.testing.assert_allclose(integral, mge.array, rtol=1e-3)
print("gNFWSph config 2: PASSED")

"""
__gNFW Elliptical Config 1__
"""

mp = ag.mp.gNFW(
    centre=(0.0, 0.0),
    kappa_s=1.0,
    ell_comps=ag.convert.ell_comps_from(axis_ratio=0.3, angle=100.0),
    inner_slope=0.5,
    scale_radius=8.0,
)
integral = deflections_2d_via_integral_from(
    mp, grid=ag.Grid2DIrregular([[0.1875, 0.1625]])
)
mge = mp.deflections_2d_via_mge_from(grid=ag.Grid2DIrregular([[0.1875, 0.1625]]))
np.testing.assert_allclose(integral, mge.array, rtol=1e-3)
print("gNFW ell config 1: PASSED")

"""
__gNFW Elliptical Config 2__
"""

mp = ag.mp.gNFW(
    centre=(0.3, 0.2),
    kappa_s=2.5,
    ell_comps=ag.convert.ell_comps_from(axis_ratio=0.5, angle=100.0),
    inner_slope=1.5,
    scale_radius=4.0,
)
integral = deflections_2d_via_integral_from(
    mp, grid=ag.Grid2DIrregular([[0.1875, 0.1625]])
)
mge = mp.deflections_2d_via_mge_from(grid=ag.Grid2DIrregular([[0.1875, 0.1625]]))
np.testing.assert_allclose(integral, mge.array, rtol=1e-3)
print("gNFW ell config 2: PASSED")

"""
__Potential — gNFWSph Inner Slope 0.5__
"""

mp = ag.mp.gNFWSph(
    centre=(0.0, 0.0), kappa_s=1.0, inner_slope=0.5, scale_radius=8.0
)
integral_potential = potential_2d_via_integral_from(
    mp, grid=ag.Grid2DIrregular([[0.1625, 0.1875]])
)
np.testing.assert_allclose(integral_potential, 0.00920, rtol=1e-3)
print("Potential inner_slope=0.5: PASSED")

"""
__Potential — gNFWSph Inner Slope 1.5__
"""

mp = ag.mp.gNFWSph(
    centre=(0.0, 0.0), kappa_s=1.0, inner_slope=1.5, scale_radius=8.0
)
integral_potential = potential_2d_via_integral_from(
    mp, grid=ag.Grid2DIrregular([[0.1625, 0.1875]])
)
np.testing.assert_allclose(integral_potential, 0.17448, rtol=1e-3)
print("Potential inner_slope=1.5: PASSED")

"""
__Potential — gNFW Elliptical__
"""

mp = ag.mp.gNFW(
    centre=(1.0, 1.0),
    kappa_s=5.0,
    ell_comps=ag.convert.ell_comps_from(axis_ratio=0.5, angle=100.0),
    inner_slope=1.0,
    scale_radius=10.0,
)
integral_potential = potential_2d_via_integral_from(
    mp, grid=ag.Grid2DIrregular([[2.0, 2.0]])
)
np.testing.assert_allclose(integral_potential, 2.4718, rtol=1e-4)
print("Potential gNFW ell: PASSED")

"""
__Convergence — gNFWSph__
"""

mp = ag.mp.gNFWSph(
    centre=(0.0, 0.0), kappa_s=1.0, inner_slope=1.5, scale_radius=1.0
)
convergence = mp.convergence_2d_from(grid=ag.Grid2DIrregular([[2.0, 0.0]]))
np.testing.assert_allclose(convergence, 0.30840, rtol=1e-2)
print("Convergence gNFWSph: PASSED")

print("\nAll gNFW integral tests passed.")
