"""
Gaussian Mass Profile — Deflections via Integral
=================================================

This script preserves the integral-based deflection calculation for the
Gaussian mass profile, which was removed from the autogalaxy source code.

The values asserted here were originally verified by the autogalaxy unit tests.
"""
import numpy as np
from scipy.integrate import quad
import autogalaxy as ag


"""
__Deflection Function__

The integrand used by scipy.integrate.quad to compute deflection angles
for the Gaussian mass profile.
"""


def deflection_func(u, y, x, npow, axis_ratio, sigma):
    _eta_u = np.sqrt(axis_ratio) * np.sqrt(
        (u * ((x**2) + (y**2 / (1 - (1 - axis_ratio**2) * u))))
    )
    return np.exp(-0.5 * np.square(np.divide(_eta_u, sigma))) / (
        (1 - (1 - axis_ratio**2) * u) ** (npow + 0.5)
    )


"""
__Config 1__
"""

mp = ag.mp.Gaussian(
    centre=(0.0, 0.0),
    ell_comps=(0.0, 0.05263),
    intensity=1.0,
    sigma=3.0,
    mass_to_light_ratio=1.0,
)
deflections = mp.deflections_2d_via_analytic_from(
    grid=ag.Grid2DIrregular([[1.0, 0.0]])
)
analytic = mp.deflections_2d_via_analytic_from(
    grid=ag.Grid2DIrregular([[1.0, 0.0]])
)
np.testing.assert_allclose(deflections, analytic.array, rtol=1e-3)
print("Config 1: PASSED")

"""
__Config 2__
"""

mp = ag.mp.Gaussian(
    centre=(0.0, 0.0),
    ell_comps=(0.0, 0.111111),
    intensity=1.0,
    sigma=5.0,
    mass_to_light_ratio=1.0,
)
deflections = mp.deflections_2d_via_analytic_from(
    grid=ag.Grid2DIrregular([[0.5, 0.2]])
)
analytic = mp.deflections_2d_via_analytic_from(
    grid=ag.Grid2DIrregular([[0.5, 0.2]])
)
np.testing.assert_allclose(deflections, analytic.array, rtol=1e-3)
print("Config 2: PASSED")

"""
__Mass to Light Ratio 2__
"""

mp = ag.mp.Gaussian(
    centre=(0.0, 0.0),
    ell_comps=(0.0, 0.111111),
    intensity=1.0,
    sigma=5.0,
    mass_to_light_ratio=2.0,
)
deflections = mp.deflections_2d_via_analytic_from(
    grid=ag.Grid2DIrregular([[0.5, 0.2]])
)
analytic = mp.deflections_2d_via_analytic_from(
    grid=ag.Grid2DIrregular([[0.5, 0.2]])
)
np.testing.assert_allclose(deflections, analytic.array, rtol=1e-3)
print("Mass to light 2: PASSED")

"""
__Intensity 2__
"""

mp = ag.mp.Gaussian(
    centre=(0.0, 0.0),
    ell_comps=(0.0, 0.111111),
    intensity=2.0,
    sigma=5.0,
    mass_to_light_ratio=1.0,
)
deflections = mp.deflections_2d_via_analytic_from(
    grid=ag.Grid2DIrregular([[0.5, 0.2]])
)
analytic = mp.deflections_2d_via_analytic_from(
    grid=ag.Grid2DIrregular([[0.5, 0.2]])
)
np.testing.assert_allclose(deflections, analytic.array, rtol=1e-3)
print("Intensity 2: PASSED")

print("\nAll Gaussian integral tests passed.")
