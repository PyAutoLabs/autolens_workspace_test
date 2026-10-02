"""
Cluster Critical Curves: Per-Source-Plane Regression
===================================================

A small deterministic CI counterpart to the critical-curve research campaign.
The research driver, timings and growing ledger belong in autolens_profiling;
this example guards the supported marching-squares geometry on two source planes.

__Contents__
Cored spherical cluster; analytic critical radii; per-plane caustic mapping.

__Env__
ENV: full_datasets
"""

import numpy as np

import autolens as al
from autogalaxy.operate.lens_calc import LensCalc


"""__Cluster__

The mass normalization is defined for the final source redshift. The nearer
source must have a smaller critical curve after distance-ratio scaling. A
finite core supplies an analytic radial critical curve without a singularity.
"""

b, core = 8.0, 0.8
tracer = al.Tracer(
    galaxies=[
        al.Galaxy(
            redshift=0.5,
            mass=al.mp.IsothermalCoreSph(einstein_radius=b, core_radius=core),
        ),
        al.Galaxy(redshift=1.0),
        al.Galaxy(redshift=2.0),
    ]
)
spacing = 0.2
grid = al.Grid2D.uniform(
    shape_native=(100, 100), pixel_scales=spacing, respect_small_datasets=False
)
radii = []
for plane_j, redshift in ((1, 1.0), (2, 2.0)):
    calc = LensCalc.from_tracer(tracer, use_multi_plane=True, plane_j=plane_j)
    scale = tracer.cosmology.scaling_factor_between_redshifts_from(
        redshift_0=0.5, redshift_1=redshift, redshift_final=2.0
    )
    effective_b = b * float(scale)
    t = (-core + np.sqrt(core**2 + 4 * effective_b * core)) / 2
    expected = {
        "tangential": np.sqrt(effective_b**2 - 2 * effective_b * core),
        "radial": np.sqrt(t**2 - core**2),
    }
    for kind in ("tangential", "radial"):
        curves = getattr(calc, kind + "_critical_curve_list_from")(
            grid=grid, pixel_scale=spacing
        )
        caustics = getattr(calc, kind + "_caustic_list_from")(
            grid=grid, pixel_scale=spacing
        )
        assert len(curves) == len(caustics) == 1, (plane_j, kind)
        points, caustic = np.asarray(curves[0]), np.asarray(caustics[0])
        assert np.isfinite(points).all() and np.isfinite(caustic).all()
        np.testing.assert_allclose(
            np.linalg.norm(points, axis=1), expected[kind], atol=spacing / 4, rtol=0
        )
        np.testing.assert_allclose(points[0], points[-1], atol=1e-10)
        mapped = points - np.asarray(calc.deflections_yx_2d_from(grid=curves[0]))
        np.testing.assert_allclose(caustic, mapped, atol=1e-10, rtol=0)
        analytic_caustic_radius = abs(
            expected[kind]
            - effective_b
            * expected[kind]
            / (np.sqrt(expected[kind] ** 2 + core**2) + core)
        )
        np.testing.assert_allclose(
            np.linalg.norm(caustic, axis=1),
            analytic_caustic_radius,
            atol=spacing / 4,
            rtol=0,
        )
        if kind == "tangential":
            radii.append(float(np.mean(np.linalg.norm(points, axis=1))))
assert radii[1] > radii[0] + spacing, radii
print("Cluster critical curves: both source planes, both kinds and caustics PASS")
