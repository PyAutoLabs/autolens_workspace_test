"""
gNFW Virial Mass Concentration — Deflections via Integral
=========================================================

This script preserves the integral-based deflection calculation for the
gNFWVirialMassConcSph mass profile, which was removed from the autogalaxy
source code.
"""
import numpy as np
import autogalaxy as ag


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
deflections = mp.deflections_2d_via_mge_from(
    grid=ag.Grid2DIrregular([[0.1875, 0.1625]])
)
np.testing.assert_allclose(deflections[0, 0], 0.0466231, rtol=1e-3)
np.testing.assert_allclose(deflections[0, 1], 0.04040671, rtol=1e-3)
print("gNFWVirialMassConcSph config 1: PASSED")

print("\nAll gNFWVirialMassConcSph integral tests passed.")
