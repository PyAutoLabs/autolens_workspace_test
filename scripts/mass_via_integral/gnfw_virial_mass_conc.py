"""
gNFW Virial Mass Concentration — Deflections via Integral
=========================================================

This script preserves the integral-based deflection calculation for the
gNFWVirialMassConcSph mass profile, which was removed from the autogalaxy
source code.
"""
import numpy as np
from scipy import special
from scipy.integrate import quad
import autogalaxy as ag


"""
__gNFWSph Deflection Integrand and Function__

Same spherical integral as gNFWSph — gNFWVirialMassConcSph inherits from it.
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
__Deflections via Integral (Spherical)__
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
__Config 1__
"""

mp = ag.mp.gNFWVirialMassConcSph(
    centre=(0.0, 0.0),
    log10m_vir=12.0,
    c_2=10.0,
    overdens=0.0,
    redshift_object=0.5,
    redshift_source=1.0,
    inner_slope=1.0,
)
integral = deflections_2d_via_integral_sph_from(
    mp, grid=ag.Grid2DIrregular([[0.1875, 0.1625]])
)
mge = mp.deflections_2d_via_mge_from(grid=ag.Grid2DIrregular([[0.1875, 0.1625]]))
np.testing.assert_allclose(integral, mge.array, rtol=1e-3)
print("gNFWVirialMassConcSph config 1: PASSED")

print("\nAll gNFWVirialMassConcSph integral tests passed.")
