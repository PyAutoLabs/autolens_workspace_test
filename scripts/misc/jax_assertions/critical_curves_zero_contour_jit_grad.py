"""
JAX: Critical Curves and Caustics via jax_zero_contour — jit and grad parity
============================================================================

``misc/critical_curves_zero_contour.py`` checks that the ``jax_zero_contour``
critical curves and caustics on ``LensCalc`` agree with the marching-squares
implementation. This script checks the two JAX transformations on the same
calculation:

 1. ``jax.jit`` parity — the tangential and radial critical curves and
    caustics computed eagerly and under an outer ``jax.jit`` (with the mass
    model built *inside* the trace, so every lens parameter is a tracer) agree
    point-for-point to 1e-10, and the jitted contours equal what the public
    eager ``LensCalc.*_via_zero_contour_from`` methods return.
 2. ``jax.grad`` correctness — the gradient of a caustic-derived scalar (the
    area enclosed by the tangential caustic, and by the radial caustic) with
    respect to the lens ``einstein_radius`` matches a central finite
    difference (the ``jax_grad/`` pattern, via ``misc/util.py``) and an exact
    analytic reference.

__Why the jit leg compares raw padded paths__

The public ``*_via_zero_contour_from`` methods end in
``ZeroSolver.path_reduce`` and convert to ``aa.Grid2DIrregular``; both have a
data-dependent output length and cannot run under ``jax.jit``. The jit-able
core — the one ``LensCalc.einstein_radius_jit_from`` already uses — is
``LensCalc._make_eigen_fn`` + ``ZeroSolver.zero_contour_finder``, which returns
a fixed-shape ``(n_seeds, 2N + 1, 2)`` path padded with NaN after the contour
closes. Both legs here return that raw array, so the comparison is exact in
shape: the NaN padding pattern must be identical (same number of points, same
ordering) and the finite entries must agree to 1e-10. No reduction or
re-ordering is needed. A separate check confirms the finite part of the
jitted array, closed by repeating its first point, is what the public eager
methods return.

__Lens model and its analytic gradient reference__

The lens is an elliptical ``PowerLaw`` (slope 1.7) plus ``ExternalShear``.
Slope < 2 is chosen deliberately: an isothermal (slope 2) lens has no radial
critical curve (it degenerates to the central point), whereas this lens has a
proper closed radial critical curve at ~0.3" and a radial caustic.

A power law with fixed slope plus a constant external shear is *scale free*:
the deflection field satisfies ``alpha(theta; b) = b * alpha(theta / b; 1)``,
so every critical curve and caustic scales exactly linearly with the Einstein
radius ``b`` and every enclosed area scales as ``b**2``. Hence

    d(area) / d(einstein_radius) = 2 * area / einstein_radius     (exact)

which gives an analytic reference independent of both autodiff and finite
differences.

__Gradient tolerances (measured 2026-11-08, jax 0.11.2, float64)__

Relative error of each estimate against the exact ``2 * area / b``:

    scalar                   autodiff    central FD (h = 1.6e-5)
    tangential caustic area  2.7e-5      5.7e-4
    radial caustic area      8.4e-5      1.1e-3

Autodiff is the *more* accurate of the two. The contour is a polygon of points
spaced ``delta`` apart along the curve, closed by one shorter "gap" edge whose
length changes as the curve grows. Finite differences differentiate that
discrete polygon, so they pick up the derivative of the sliver of area under
the gap edge (~ curvature * gap**2 * d(gap)/db / 4, i.e. O(delta**2) — this
estimate reproduces the observed FD offset). Autodiff differentiates the
Newton projection of each point through ``jax.lax.custom_root`` (implicit
function theorem), which moves points normal to the curve and so tracks the
continuous area. The FD values are converged in step size (identical to 7
significant figures for h = 1e-4 .. 1e-6), so the offset is the discretisation
sliver, not FD noise.

The tolerances therefore are:

 - autodiff vs finite differences: ``rtol = 3e-3`` (worst measured 1.2e-3).
 - autodiff vs exact ``2 * area / b``: ``rtol = 5e-4`` (worst measured
   8.4e-5) — the tight, falsifiable check. A gradient that dropped a term
   (e.g. the explicit ``-alpha`` dependence of the caustic, or the
   critical-curve motion) would be off by O(1), not O(1e-4).
 - eager ``jax.grad`` vs jitted ``jax.grad``: 1e-10.

__Env__

Test-harness configuration (PyAutoHands docs/env_profile_redesign.md §10).
Runs JAX transformations directly on ``LensCalc`` (no search, no dataset);
float64 is forced in-script below.

ENV: jax
"""

import os
import sys
import time

import numpy as np
import numpy.testing as npt
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

import autolens as al
from autogalaxy.operate._zero_contour import zero_solver_type

# ``util.py`` (the shared finite-difference gradient helper) lives in
# ``scripts/misc/``, one level up from this script.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import util

ZeroSolver = zero_solver_type()

"""
__Setup__

Fixed contour-tracing settings. Seeds sit near each curve; ``N`` comfortably
exceeds the number of steps either closed curve needs (~215 tangential, ~95
radial), so both contours close before the step budget runs out.
"""
einstein_radius = 1.6

N = 200
tol = 1e-6
max_newton = 5

settings = {
    "tangential": {"seed": jnp.array([[1.6, 0.0]]), "delta": 0.05},
    "radial": {"seed": jnp.array([[0.3, 0.0]]), "delta": 0.02},
}


def tracer_from(einstein_radius):
    lens = al.Galaxy(
        redshift=0.5,
        mass=al.mp.PowerLaw(
            centre=(0.0, 0.0),
            ell_comps=(0.1, 0.05),
            einstein_radius=einstein_radius,
            slope=1.7,
        ),
        shear=al.mp.ExternalShear(gamma_1=0.05, gamma_2=-0.02),
    )
    return al.Tracer(galaxies=[lens, al.Galaxy(redshift=1.0)])


def contours_from(einstein_radius, kind):
    """
    Critical curve and caustic of ``kind`` as raw NaN-padded arrays of shape
    ``(1, 2N + 1, 2)``, plus the ``ZeroSolver`` stopping conditions.

    Mirrors ``LensCalc._critical_curve_list_via_zero_contour`` and
    ``_caustic_list_via_zero_contour`` (same eigen function, solver and
    ``caustic = cc - deflections(cc)``) minus the non-jit-able
    ``path_reduce`` / ``Grid2DIrregular`` conversion. The tracer is built from
    ``einstein_radius`` inside this function so that, under ``jax.jit`` or
    ``jax.grad``, the whole mass model is traced.
    """
    lens_calc = al.LensCalc.from_tracer(tracer=tracer_from(einstein_radius))

    f = lens_calc._make_eigen_fn(kind=kind)
    solver = ZeroSolver(tol=tol, max_newton=max_newton)

    paths, stop = solver.zero_contour_finder(
        f, settings[kind]["seed"], delta=settings[kind]["delta"], N=N
    )
    critical_curve = paths["path"]

    points = critical_curve.reshape(-1, 2)
    finite = jnp.isfinite(points)
    # Ray-trace the padding rows at a harmless finite point, then restore NaN,
    # so no NaN enters the deflection calculation (or its cotangents).
    grid = al.Grid2DIrregular(values=jnp.where(finite, points, 0.0))
    deflections = lens_calc.deflections_yx_2d_from(grid=grid, xp=jnp)
    deflections = getattr(deflections, "array", deflections)
    caustic = jnp.where(finite, points - deflections, jnp.nan).reshape(
        critical_curve.shape
    )

    return critical_curve, caustic, stop


def closed_area_from(curve):
    """
    Shoelace area of one NaN-padded contour ``(2N + 1, 2)``. Padding rows are
    replaced by the first point, so the polygon is closed by exactly one edge
    (last finite point -> first point) and the padding adds zero area. The
    mask is a function of the contour's NaN pattern only, so no NaN reaches
    the gradient.
    """
    finite = jnp.isfinite(curve[:, 0])
    closed = jnp.where(finite[:, None], curve, curve[0])
    nxt = jnp.roll(closed, -1, axis=0)
    return 0.5 * jnp.abs(jnp.sum(closed[:, 1] * nxt[:, 0] - nxt[:, 1] * closed[:, 0]))


def finite_part(curve):
    curve = np.asarray(curve)
    return curve[np.isfinite(curve).all(axis=1)]


"""
__jax.jit parity__

Evaluate each kind eagerly and under ``jax.jit`` and compare. The eager leg
still runs ``zero_contour_finder`` compiled (it is ``jax.jit``-decorated
upstream); what the outer ``jax.jit`` adds is tracing the mass model, the
eigen function, the solver call and the caustic ray-trace as one program with
a traced ``einstein_radius``.
"""
contours_jit = jax.jit(contours_from, static_argnums=(1,))

eager = {}
max_jit_diff = {}

for kind in ("tangential", "radial"):
    start = time.time()
    cc_eager, ca_eager, stop_eager = contours_from(einstein_radius, kind)
    cc_jit, ca_jit, stop_jit = contours_jit(einstein_radius, kind)
    eager[kind] = (cc_eager, ca_eager)

    n_points = int(np.isfinite(np.asarray(cc_eager)[0, :, 0]).sum())
    assert n_points > 20, f"{kind}: only {n_points} contour points traced"
    assert 2 in np.asarray(stop_eager), (
        f"{kind}: contour did not close (stopping conditions {stop_eager}); "
        "increase N or move the seed."
    )
    npt.assert_array_equal(np.asarray(stop_eager), np.asarray(stop_jit))

    for name, a, b in (
        ("critical curve", cc_eager, cc_jit),
        ("caustic", ca_eager, ca_jit),
    ):
        a, b = np.asarray(a), np.asarray(b)
        npt.assert_array_equal(
            np.isfinite(a),
            np.isfinite(b),
            err_msg=f"{kind} {name}: eager and jit NaN padding differ "
            "(different number / ordering of contour points)",
        )
        mask = np.isfinite(a)
        diff = float(np.max(np.abs(a[mask] - b[mask])))
        max_jit_diff[f"{kind} {name}"] = diff
        npt.assert_allclose(
            b[mask],
            a[mask],
            rtol=0.0,
            atol=1e-10,
            err_msg=f"{kind} {name}: jax.jit differs from eager",
        )

    print(
        f"{kind:<10}  points={n_points:>4}  "
        f"max|jit - eager| cc={max_jit_diff[f'{kind} critical curve']:.3e}  "
        f"caustic={max_jit_diff[f'{kind} caustic']:.3e}  ({time.time() - start:.1f}s)"
    )

"""
__Public eager API vs jitted contours__

The public methods return the same contour, closed by repeating its first
point, as a list of ``Grid2DIrregular``. Their caustics use the NumPy
deflection path, so this also checks NumPy-vs-JAX deflections along the curve.
"""
lens_calc = al.LensCalc.from_tracer(tracer=tracer_from(einstein_radius))

public = {
    "tangential": (
        lens_calc.tangential_critical_curve_list_via_zero_contour_from,
        lens_calc.tangential_caustic_list_via_zero_contour_from,
    ),
    "radial": (
        lens_calc.radial_critical_curve_list_via_zero_contour_from,
        lens_calc.radial_caustic_list_via_zero_contour_from,
    ),
}

for kind, (cc_method, ca_method) in public.items():
    kwargs = dict(
        init_guess=settings[kind]["seed"],
        delta=settings[kind]["delta"],
        N=N,
        tol=tol,
        max_newton=max_newton,
    )
    cc_jit, ca_jit, _ = contours_jit(einstein_radius, kind)

    for name, method, raw in (
        ("critical curve", cc_method, cc_jit),
        ("caustic", ca_method, ca_jit),
    ):
        curves = method(**kwargs)
        assert len(curves) == 1, f"public {kind} {name}: {len(curves)} curves"
        expected = finite_part(raw[0])
        expected = np.vstack([expected, expected[0]])
        npt.assert_allclose(
            np.asarray(curves[0]),
            expected,
            rtol=0.0,
            atol=1e-10,
            err_msg=f"public eager {kind} {name} differs from the jitted contour",
        )

print("public *_via_zero_contour_from methods match the jitted contours to 1e-10")

"""
__jax.grad of caustic areas__

For each caustic: ``jax.grad`` of the enclosed area w.r.t. ``einstein_radius``
— eager and jitted — compared with central finite differences (via
``util.compare_gradients``, FD on the jitted function after the
eager-vs-jit consistency guard) and with the exact scale-free reference
``2 * area / einstein_radius`` (see the module docstring for the tolerances).
"""
x = jnp.array([einstein_radius])

for kind in ("tangential", "radial"):
    start = time.time()

    def caustic_area(params, kind=kind):
        _, caustic, _ = contours_from(params[0], kind)
        return closed_area_from(caustic[0])

    caustic_area_jit = jax.jit(caustic_area)

    util.assert_eager_jit_consistent(caustic_area, caustic_area_jit, x, rtol=1e-10)

    comparison = util.compare_gradients(
        caustic_area,
        x,
        param_names=[f"d({kind} caustic area)/d(einstein_radius)"],
        f_fd=caustic_area_jit,
    )
    util.assert_gradients_match(comparison, rtol=3e-3, atol=0.0)

    grad_eager = float(comparison["ad"][0])
    grad_jit = float(jax.jit(jax.grad(caustic_area))(x)[0])
    npt.assert_allclose(
        grad_jit,
        grad_eager,
        rtol=1e-10,
        err_msg=f"{kind} caustic area: jitted jax.grad differs from eager jax.grad",
    )

    area = float(caustic_area_jit(x))
    exact = 2.0 * area / einstein_radius
    print(
        f"{kind} caustic: area={area:.10f}  exact 2A/b={exact:.10f}  "
        f"rel err autodiff={abs(grad_eager - exact) / exact:.2e}  "
        f"FD={abs(comparison['fd'][0] - exact) / exact:.2e}  "
        f"|grad_jit - grad_eager|={abs(grad_jit - grad_eager):.2e}  "
        f"({time.time() - start:.1f}s)"
    )
    npt.assert_allclose(
        grad_eager,
        exact,
        rtol=5e-4,
        err_msg=f"{kind} caustic area: jax.grad disagrees with exact 2 * area / b",
    )

print("All critical_curves_zero_contour_jit_grad tests passed.")
