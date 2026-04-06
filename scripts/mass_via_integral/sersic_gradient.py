"""
Sersic Gradient Mass Profile — Deflections via Integral
=======================================================

This script preserves the integral-based deflection calculation for the
SersicGradient mass profile, which was removed from the autogalaxy source code.
"""
import numpy as np
from scipy.integrate import quad
import autogalaxy as ag


"""
__Deflection Function__
"""


def deflection_func(
    u,
    y,
    x,
    npow,
    axis_ratio,
    sersic_index,
    effective_radius,
    mass_to_light_gradient,
    sersic_constant,
):
    _eta_u = np.sqrt(axis_ratio) * np.sqrt(
        (u * ((x**2) + (y**2 / (1 - (1 - axis_ratio**2) * u))))
    )
    return (
        (((axis_ratio * _eta_u) / effective_radius) ** -mass_to_light_gradient)
        * np.exp(
            -sersic_constant
            * (((_eta_u / effective_radius) ** (1.0 / sersic_index)) - 1)
        )
        / ((1 - (1 - axis_ratio**2) * u) ** (npow + 0.5))
    )


"""
__Gradient 1__
"""

mp = ag.mp.SersicGradient(
    centre=(-0.4, -0.2),
    ell_comps=(-0.07142, -0.085116),
    intensity=5.0,
    effective_radius=0.2,
    sersic_index=2.0,
    mass_to_light_ratio=1.0,
    mass_to_light_gradient=1.0,
)
deflections = mp.deflections_2d_via_integral_from(
    grid=ag.Grid2DIrregular([[0.1625, 0.1625]])
)
np.testing.assert_allclose(deflections[0, 0], 3.60324873535244, rtol=1e-3)
np.testing.assert_allclose(deflections[0, 1], 2.3638898009652, rtol=1e-3)
print("Gradient 1: PASSED")

"""
__Gradient -1__
"""

mp = ag.mp.SersicGradient(
    centre=(-0.4, -0.2),
    ell_comps=(-0.07142, -0.085116),
    intensity=5.0,
    effective_radius=0.2,
    sersic_index=2.0,
    mass_to_light_ratio=1.0,
    mass_to_light_gradient=-1.0,
)
deflections = mp.deflections_2d_via_integral_from(
    grid=ag.Grid2DIrregular([[0.1625, 0.1625]])
)
np.testing.assert_allclose(deflections[0, 0], 0.97806399756448, rtol=1e-3)
np.testing.assert_allclose(deflections[0, 1], 0.725459334118341, rtol=1e-3)
print("Gradient -1: PASSED")

print("\nAll SersicGradient integral tests passed.")
