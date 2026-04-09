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
__Deflections via Integral__
"""


def deflections_2d_via_integral_from(mp, grid):
    transformed = mp.transformed_to_reference_frame_grid_from(grid)
    axis_ratio = mp.axis_ratio()
    sersic_constant = mp.sersic_constant

    def calculate_deflection_component(npow, index):
        deflection_grid = axis_ratio * transformed.array[:, index]

        for i in range(transformed.shape[0]):
            deflection_grid[i] = deflection_grid[i] * (
                mp.intensity
                * mp.mass_to_light_ratio
                * quad(
                    deflection_func,
                    a=0.0,
                    b=1.0,
                    args=(
                        transformed.array[i, 0],
                        transformed.array[i, 1],
                        npow,
                        axis_ratio,
                        mp.sersic_index,
                        mp.effective_radius,
                        sersic_constant,
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
integral = deflections_2d_via_integral_from(
    mp, grid=ag.Grid2DIrregular([[0.1625, 0.1625]])
)
cse = mp.deflections_2d_via_cse_from(grid=ag.Grid2DIrregular([[0.1625, 0.1625]]))
np.testing.assert_allclose(integral, cse.array, rtol=1e-3)
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
integral = deflections_2d_via_integral_from(
    mp, grid=ag.Grid2DIrregular([[0.1625, 0.1625]])
)
cse = mp.deflections_2d_via_cse_from(grid=ag.Grid2DIrregular([[0.1625, 0.1625]]))
np.testing.assert_allclose(integral, cse.array, rtol=1e-3)
print("Config 2: PASSED")

print("\nAll Sersic integral tests passed.")
