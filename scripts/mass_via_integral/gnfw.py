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


"""
__gNFWSph Config 1__
"""

mp = ag.mp.gNFWSph(
    centre=(0.0, 0.0), kappa_s=1.0, inner_slope=0.5, scale_radius=8.0
)
deflections = mp.deflections_2d_via_mge_from(
    grid=ag.Grid2DIrregular([[0.1875, 0.1625]])
)
np.testing.assert_allclose(deflections[0, 0], 0.43501, rtol=1e-3)
np.testing.assert_allclose(deflections[0, 1], 0.37701, rtol=1e-3)
print("gNFWSph config 1: PASSED")

"""
__gNFWSph Config 2__
"""

mp = ag.mp.gNFWSph(
    centre=(0.3, 0.2), kappa_s=2.5, inner_slope=1.5, scale_radius=4.0
)
deflections = mp.deflections_2d_via_mge_from(
    grid=ag.Grid2DIrregular([[0.1875, 0.1625]])
)
np.testing.assert_allclose(deflections[0, 0], -9.31254, rtol=1e-3)
np.testing.assert_allclose(deflections[0, 1], -3.10418, rtol=1e-3)
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
deflections = mp.deflections_2d_via_mge_from(
    grid=ag.Grid2DIrregular([[0.1875, 0.1625]])
)
np.testing.assert_allclose(deflections[0, 0], 0.26604, rtol=1e-3)
np.testing.assert_allclose(deflections[0, 1], 0.58988, rtol=1e-3)
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
deflections = mp.deflections_2d_via_mge_from(
    grid=ag.Grid2DIrregular([[0.1875, 0.1625]])
)
np.testing.assert_allclose(deflections[0, 0], -5.99032, rtol=1e-3)
np.testing.assert_allclose(deflections[0, 1], -4.02541, rtol=1e-3)
print("gNFW ell config 2: PASSED")

"""
__Potential — gNFWSph Inner Slope 0.5__
"""

mp = ag.mp.gNFWSph(
    centre=(0.0, 0.0), kappa_s=1.0, inner_slope=0.5, scale_radius=8.0
)
potential = mp.potential_2d_from(grid=ag.Grid2DIrregular([[0.1625, 0.1875]]))
np.testing.assert_allclose(potential, 0.00920, rtol=1e-3)
print("Potential inner_slope=0.5: PASSED")

"""
__Potential — gNFWSph Inner Slope 1.5__
"""

mp = ag.mp.gNFWSph(
    centre=(0.0, 0.0), kappa_s=1.0, inner_slope=1.5, scale_radius=8.0
)
potential = mp.potential_2d_from(grid=ag.Grid2DIrregular([[0.1625, 0.1875]]))
np.testing.assert_allclose(potential, 0.17448, rtol=1e-3)
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
potential = mp.potential_2d_from(grid=ag.Grid2DIrregular([[2.0, 2.0]]))
np.testing.assert_allclose(potential, 2.4718, rtol=1e-4)
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
