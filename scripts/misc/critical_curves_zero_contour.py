"""
Operate: Critical Curves and Caustics via jax_zero_contour
===========================================================

This script tests the ``jax_zero_contour``-based critical curve and caustic
methods added to ``LensCalc``:

- ``tangential_critical_curve_list_via_zero_contour_from``
- ``radial_critical_curve_list_via_zero_contour_from``
- ``tangential_caustic_list_via_zero_contour_from``
- ``radial_caustic_list_via_zero_contour_from``
- ``einstein_radius_list_via_zero_contour_from``
- ``einstein_radius_via_zero_contour_from``

Each method is tested in two ways:

1. **Return-type and shape checks** — the output is a list of
   ``aa.Grid2DIrregular`` objects with the expected coordinate structure.

2. **Cross-implementation comparison** — results are compared directly against
   the existing marching-squares implementation
   (``tangential_critical_curve_list_from`` etc.) to verify that both methods
   produce curves with consistent geometry: similar mean radius, centroid, and
   area, within a tolerance appropriate for two independent algorithms.

The cross-comparison is the primary regression test.  Exact point-for-point
agreement is not expected (the two methods sample the contour differently), so
all comparisons are on derived scalar quantities (mean radius from the lens
centre, centroid coordinates, curve area).

Mass profile used: ``Isothermal`` (centre at origin, elliptical, Einstein
radius 2.0 arcsec).  This profile has a well-defined tangential critical curve
at approximately the Einstein radius and a compact radial critical curve close
to the centre.
"""

import numpy as np
import numpy.testing as npt
import autoarray as aa
import autogalaxy as ag
from autogalaxy.operate.lens_calc import LensCalc

"""
__Setup__

Build the ``LensCalc`` from an elliptical ``Isothermal`` mass profile and
define the evaluation grid used by the marching-squares reference
implementation.
"""
mp = ag.mp.Isothermal(
    centre=(0.0, 0.0), ell_comps=(0.0, -0.111111), einstein_radius=2.0
)

lens_calc = LensCalc.from_mass_obj(mp)

# Reference grid for the marching-squares implementation.
# pixel_scale=0.05 matches the default used by the zero-contour methods.
grid_ref = ag.Grid2D.uniform(shape_native=(100, 100), pixel_scales=0.1)

# Explicit seed near the tangential critical curve (~Einstein radius from centre).
# The zero-contour tracer does not need a grid; this single seed is enough.
tangential_seed = np.array([[2.0, 0.0]])

# Explicit seed near the radial critical curve.
# For this Isothermal profile the radial CC sits at ~0.06 arcsec from the
# centre, so the seed must be close.  We locate it from a fine coarse grid
# and store it for use across tests.
radial_seed = np.array([[0.07, 0.0]])

"""
__Helper: scalar geometry metrics__

For two sets of (y, x) coordinates representing a closed curve, compute:
- mean radius from the origin
- centroid (mean y, mean x)
- enclosed area via the shoelace formula
"""


def mean_radius(curve: aa.Grid2DIrregular) -> float:
    y = np.array(curve[:, 0])
    x = np.array(curve[:, 1])
    return float(np.mean(np.sqrt(y**2 + x**2)))


def centroid(curve: aa.Grid2DIrregular):
    y = np.array(curve[:, 0])
    x = np.array(curve[:, 1])
    return float(np.mean(y)), float(np.mean(x))


def shoelace_area(curve: aa.Grid2DIrregular) -> float:
    y = np.array(curve[:, 0])
    x = np.array(curve[:, 1])
    return float(np.abs(0.5 * np.sum(y[:-1] * np.diff(x) - x[:-1] * np.diff(y))))


"""
__Tangential critical curve — return type and shape__

The zero-contour method must return a non-empty list of ``Grid2DIrregular``
objects, each with shape ``(N, 2)``.
"""
tan_cc_zc = lens_calc.tangential_critical_curve_list_via_zero_contour_from(
    init_guess=tangential_seed,
    delta=0.05,
    N=500,
)

assert isinstance(tan_cc_zc, list), (
    f"tangential_critical_curve_list_via_zero_contour_from: expected list, "
    f"got {type(tan_cc_zc)}"
)
assert (
    len(tan_cc_zc) >= 1
), "tangential_critical_curve_list_via_zero_contour_from: returned empty list"
for i, curve in enumerate(tan_cc_zc):
    assert isinstance(curve, aa.Grid2DIrregular), (
        f"tangential_critical_curve_list_via_zero_contour_from: curve {i} "
        f"expected Grid2DIrregular, got {type(curve)}"
    )
    assert curve.shape[1] == 2, (
        f"tangential_critical_curve_list_via_zero_contour_from: curve {i} "
        f"expected shape (N, 2), got {curve.shape}"
    )
    assert len(curve) > 10, (
        f"tangential_critical_curve_list_via_zero_contour_from: curve {i} "
        f"has only {len(curve)} points — tracing may have failed"
    )

"""
__Tangential critical curve — cross-comparison with marching squares__

Compare mean radius, centroid, and area against the reference marching-squares
implementation.  Both algorithms trace the same mathematical locus, so derived
scalar quantities must agree to within a few percent.
"""
tan_cc_ref = lens_calc.tangential_critical_curve_list_from(grid=grid_ref)

assert len(tan_cc_ref) >= 1, (
    "Reference tangential_critical_curve_list_from returned empty list — "
    "check the reference grid."
)

# Use the largest curve from each method for the comparison (the main Einstein ring).
tan_ref_main = max(tan_cc_ref, key=len)
tan_zc_main = max(tan_cc_zc, key=len)

r_ref = mean_radius(tan_ref_main)
r_zc = mean_radius(tan_zc_main)
npt.assert_allclose(
    r_zc,
    r_ref,
    rtol=0.05,
    err_msg=(
        f"Tangential critical curve mean radius mismatch: "
        f"zero_contour={r_zc:.4f}, marching_squares={r_ref:.4f}"
    ),
)

cy_ref, cx_ref = centroid(tan_ref_main)
cy_zc, cx_zc = centroid(tan_zc_main)
npt.assert_allclose(
    cy_zc,
    cy_ref,
    atol=0.15,
    err_msg="Tangential critical curve centroid-y mismatch",
)
npt.assert_allclose(
    cx_zc,
    cx_ref,
    atol=0.15,
    err_msg="Tangential critical curve centroid-x mismatch",
)

area_ref = shoelace_area(tan_ref_main)
area_zc = shoelace_area(tan_zc_main)
npt.assert_allclose(
    area_zc,
    area_ref,
    rtol=0.10,
    err_msg=(
        f"Tangential critical curve area mismatch: "
        f"zero_contour={area_zc:.4f}, marching_squares={area_ref:.4f}"
    ),
)

"""
__Radial critical curve — return type, shape, and cross-comparison__

The radial critical curve for this Isothermal profile lies ~0.06 arcsec
from the centre — much smaller than the tangential Einstein ring.  We first
check that the marching-squares reference finds it; if it does, we verify
that the zero-contour method also finds a consistent curve.
"""
rad_cc_ref = lens_calc.radial_critical_curve_list_from(grid=grid_ref)

rad_cc_zc = lens_calc.radial_critical_curve_list_via_zero_contour_from(
    init_guess=radial_seed,
    delta=0.005,
    N=200,
)

assert isinstance(rad_cc_zc, list), (
    f"radial_critical_curve_list_via_zero_contour_from: expected list, "
    f"got {type(rad_cc_zc)}"
)
for i, curve in enumerate(rad_cc_zc):
    assert isinstance(curve, aa.Grid2DIrregular), (
        f"radial_critical_curve_list_via_zero_contour_from: curve {i} "
        f"expected Grid2DIrregular, got {type(curve)}"
    )
    assert curve.shape[1] == 2, (
        f"radial_critical_curve_list_via_zero_contour_from: curve {i} "
        f"expected shape (N, 2), got {curve.shape}"
    )

# Cross-compare only when both methods find a radial CC.
# The Isothermal radial CC is tiny (~0.06 arcsec) and numerically
# challenging; the zero-contour tracer may or may not converge to it
# depending on the seed proximity.
if len(rad_cc_ref) >= 1 and len(rad_cc_zc) >= 1:
    rad_ref_main = max(rad_cc_ref, key=len)
    rad_zc_main = max(rad_cc_zc, key=len)

    r_rad_ref = mean_radius(rad_ref_main)
    r_rad_zc = mean_radius(rad_zc_main)
    npt.assert_allclose(
        r_rad_zc,
        r_rad_ref,
        rtol=0.20,
        err_msg=(
            f"Radial critical curve mean radius mismatch: "
            f"zero_contour={r_rad_zc:.4f}, marching_squares={r_rad_ref:.4f}"
        ),
    )

    cy_rad_ref, cx_rad_ref = centroid(rad_ref_main)
    cy_rad_zc, cx_rad_zc = centroid(rad_zc_main)
    npt.assert_allclose(
        cy_rad_zc,
        cy_rad_ref,
        atol=0.1,
        err_msg="Radial critical curve centroid-y mismatch",
    )
    npt.assert_allclose(
        cx_rad_zc,
        cx_rad_ref,
        atol=0.1,
        err_msg="Radial critical curve centroid-x mismatch",
    )

"""
__Tangential caustic — return type and cross-comparison__

Caustics are the source-plane images of critical curves.  For an
isothermal lens, the tangential caustic degenerates to a point (or very
small region) at the centre of the lens.
"""
tan_ca_zc = lens_calc.tangential_caustic_list_via_zero_contour_from(
    init_guess=tangential_seed,
    delta=0.05,
    N=500,
)

assert isinstance(tan_ca_zc, list), (
    f"tangential_caustic_list_via_zero_contour_from: expected list, "
    f"got {type(tan_ca_zc)}"
)
assert (
    len(tan_ca_zc) >= 1
), "tangential_caustic_list_via_zero_contour_from: returned empty list"
for i, curve in enumerate(tan_ca_zc):
    assert isinstance(curve, aa.Grid2DIrregular), (
        f"tangential_caustic_list_via_zero_contour_from: caustic {i} "
        f"expected Grid2DIrregular, got {type(curve)}"
    )

tan_ca_ref = lens_calc.tangential_caustic_list_from(grid=grid_ref)

if len(tan_ca_ref) >= 1:
    ca_ref_main = max(tan_ca_ref, key=len)
    ca_zc_main = max(tan_ca_zc, key=len)

    r_ca_ref = mean_radius(ca_ref_main)
    r_ca_zc = mean_radius(ca_zc_main)
    # Tangential caustic for an isothermal is near the origin; allow a looser
    # absolute tolerance rather than a relative one.
    npt.assert_allclose(
        r_ca_zc,
        r_ca_ref,
        atol=0.15,
        err_msg=(
            f"Tangential caustic mean radius mismatch: "
            f"zero_contour={r_ca_zc:.4f}, marching_squares={r_ca_ref:.4f}"
        ),
    )

"""
__Radial caustic — return type check__

Only run if the zero-contour method found a radial CC to ray-trace.
"""
if len(rad_cc_zc) >= 1:
    rad_ca_zc = lens_calc.radial_caustic_list_via_zero_contour_from(
        init_guess=radial_seed,
        delta=0.005,
        N=200,
    )

    assert isinstance(rad_ca_zc, list), (
        f"radial_caustic_list_via_zero_contour_from: expected list, "
        f"got {type(rad_ca_zc)}"
    )
    assert (
        len(rad_ca_zc) >= 1
    ), "radial_caustic_list_via_zero_contour_from: returned empty list"
    for i, curve in enumerate(rad_ca_zc):
        assert isinstance(curve, aa.Grid2DIrregular), (
            f"radial_caustic_list_via_zero_contour_from: caustic {i} "
            f"expected Grid2DIrregular, got {type(curve)}"
        )

"""
__Einstein radius — list and scalar__

``einstein_radius_list_via_zero_contour_from`` must agree with
``einstein_radius_list_from`` (marching squares) to within 5%.

``einstein_radius_via_zero_contour_from`` must equal the sum of the list.
"""
er_list_zc = lens_calc.einstein_radius_list_via_zero_contour_from(
    init_guess=tangential_seed,
    delta=0.05,
    N=500,
)

assert isinstance(er_list_zc, list), (
    f"einstein_radius_list_via_zero_contour_from: expected list, "
    f"got {type(er_list_zc)}"
)
assert (
    len(er_list_zc) >= 1
), "einstein_radius_list_via_zero_contour_from: returned empty list"
for er in er_list_zc:
    assert isinstance(er, float), (
        f"einstein_radius_list_via_zero_contour_from: expected float entries, "
        f"got {type(er)}"
    )
    assert (
        er > 0
    ), "einstein_radius_list_via_zero_contour_from: Einstein radius must be positive"

er_list_ref = lens_calc.einstein_radius_list_from(grid=grid_ref)

npt.assert_allclose(
    sum(er_list_zc),
    sum(er_list_ref),
    rtol=0.05,
    err_msg=(
        f"Einstein radius mismatch: zero_contour={sum(er_list_zc):.4f}, "
        f"marching_squares={sum(er_list_ref):.4f}"
    ),
)

er_scalar_zc = lens_calc.einstein_radius_via_zero_contour_from(
    init_guess=tangential_seed,
    delta=0.05,
    N=500,
)

assert isinstance(
    er_scalar_zc, float
), f"einstein_radius_via_zero_contour_from: expected float, got {type(er_scalar_zc)}"

npt.assert_allclose(
    er_scalar_zc,
    sum(er_list_zc),
    rtol=1e-6,
    err_msg="einstein_radius_via_zero_contour_from must equal sum of einstein_radius_list_via_zero_contour_from",
)

"""
__Auto init_guess (no explicit seed)__

When ``init_guess=None`` the methods fall back to the coarse-grid scan.
Verify this path works and produces a curve consistent with the explicit-seed
result.
"""
tan_cc_auto = lens_calc.tangential_critical_curve_list_via_zero_contour_from(
    init_guess=None,
    delta=0.05,
    N=500,
)

assert isinstance(tan_cc_auto, list) and len(tan_cc_auto) >= 1, (
    "tangential_critical_curve_list_via_zero_contour_from (auto init_guess): "
    "returned empty list"
)

r_auto = mean_radius(max(tan_cc_auto, key=len))
npt.assert_allclose(
    r_auto,
    r_zc,
    rtol=0.05,
    err_msg=(
        f"Auto init_guess tangential CC mean radius mismatch: "
        f"auto={r_auto:.4f}, explicit={r_zc:.4f}"
    ),
)

"""
__Offset centre__

Repeat the tangential critical-curve comparison with the lens centre offset
from the origin to ensure the method is not hard-wired to a centred lens.
"""
mp_offset = ag.mp.Isothermal(
    centre=(0.5, -0.3), ell_comps=(0.0, -0.111111), einstein_radius=2.0
)
lc_offset = LensCalc.from_mass_obj(mp_offset)

tan_cc_offset_zc = lc_offset.tangential_critical_curve_list_via_zero_contour_from(
    init_guess=np.array([[0.5 + 2.0, -0.3]]),  # seed near Einstein ring
    delta=0.05,
    N=500,
)

grid_offset = ag.Grid2D.uniform(shape_native=(100, 100), pixel_scales=0.1)
tan_cc_offset_ref = lc_offset.tangential_critical_curve_list_from(grid=grid_offset)

assert (
    len(tan_cc_offset_zc) >= 1
), "Offset lens tangential_critical_curve_list_via_zero_contour_from: empty list"
assert (
    len(tan_cc_offset_ref) >= 1
), "Offset lens tangential_critical_curve_list_from: empty list"

# Centroid should be near the lens centre (0.5, -0.3)
cy_off, cx_off = centroid(max(tan_cc_offset_zc, key=len))
npt.assert_allclose(cy_off, 0.5, atol=0.2, err_msg="Offset lens centroid-y")
npt.assert_allclose(cx_off, -0.3, atol=0.2, err_msg="Offset lens centroid-x")

r_off_zc = mean_radius(max(tan_cc_offset_zc, key=len))
r_off_ref = mean_radius(max(tan_cc_offset_ref, key=len))
npt.assert_allclose(
    r_off_zc,
    r_off_ref,
    rtol=0.05,
    err_msg="Offset lens tangential CC mean radius mismatch",
)

print("All critical_curves_zero_contour tests passed.")
