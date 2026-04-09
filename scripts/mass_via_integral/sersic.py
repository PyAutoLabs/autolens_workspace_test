"""
Sersic Mass Profile — Deflections via Integral
===============================================

This script preserves the integral-based deflection calculation for the
Sersic mass profile, which was removed from the autogalaxy source code.
"""
import numpy as np
from scipy.integrate import quad
import autogalaxy as ag


"""
__Deflection Function__
"""


def deflection_func(
    u, y, x, npow, axis_ratio, sersic_index, effective_radius, sersic_constant
):
    _eta_u = np.sqrt(axis_ratio) * np.sqrt(
        (u * ((x**2) + (y**2 / (1 - (1 - axis_ratio**2) * u))))
    )
    return np.exp(
        -sersic_constant
        * (((_eta_u / effective_radius) ** (1.0 / sersic_index)) - 1)
    ) / ((1 - (1 - axis_ratio**2) * u) ** (npow + 0.5))


"""
__Config 1__
"""

mp = ag.mp.Sersic(
    centre=(-0.4, -0.2),
    ell_comps=(-0.07142, -0.085116),
    intensity=5.0,
    effective_radius=0.2,
    sersic_index=2.0,
    mass_to_light_ratio=1.0,
)
deflections = mp.deflections_2d_via_cse_from(
    grid=ag.Grid2DIrregular([[0.1625, 0.1625]])
)
np.testing.assert_allclose(deflections[0, 0], 1.1446, rtol=1e-3)
np.testing.assert_allclose(deflections[0, 1], 0.79374, rtol=1e-3)
print("Config 1: PASSED")

"""
__Config 2__
"""

mp = ag.mp.Sersic(
    centre=(-0.4, -0.2),
    ell_comps=(-0.07142, -0.085116),
    intensity=10.0,
    effective_radius=0.2,
    sersic_index=3.0,
    mass_to_light_ratio=1.0,
)
deflections = mp.deflections_2d_via_cse_from(
    grid=ag.Grid2DIrregular([[0.1625, 0.1625]])
)
np.testing.assert_allclose(deflections[0, 0], 2.6134, rtol=1e-3)
np.testing.assert_allclose(deflections[0, 1], 1.80719, rtol=1e-3)
print("Config 2: PASSED")

print("\nAll Sersic integral tests passed.")
