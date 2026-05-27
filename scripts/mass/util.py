"""
Mass Profile Self-Consistency Utilities
=======================================

Shared numerical differentiation helpers and lensing relation checks used by
all per-category mass profile test scripts.

Set ``PYAUTO_MASS_FAST=1`` for smaller grids and looser tolerances.
"""

import os
import numpy as np
import autogalaxy as ag

FAST = os.environ.get("PYAUTO_MASS_FAST", "0") == "1"


def make_grid():
    shape = (40, 40) if FAST else (100, 100)
    return ag.Grid2D.uniform(shape_native=shape, pixel_scales=0.05)


def get_tolerances():
    return dict(rtol=5e-2) if FAST else dict(rtol=1e-2)


def trim_border(arr, n=3):
    return arr[n:-n, n:-n]


def _to_native_2d(result, grid):
    if hasattr(result, "native"):
        return result.native
    arr = np.asarray(result)
    if arr.ndim == 1:
        return arr.reshape(grid.shape_native)
    return arr


def _to_native_vector(result, grid):
    if hasattr(result, "native"):
        return result.native[:, :, 0], result.native[:, :, 1]
    arr = np.asarray(result)
    if arr.ndim == 2 and arr.shape[1] == 2:
        vy = arr[:, 0].reshape(grid.shape_native)
        vx = arr[:, 1].reshape(grid.shape_native)
        return vy, vx
    return arr[:, :, 0], arr[:, :, 1]


def divergence_2d(vy, vx, dx):
    # y decreases along axis 0 in autoarray native grids → negative spacing
    dvy_dy = np.gradient(vy, -dx, axis=0)
    dvx_dx = np.gradient(vx, dx, axis=1)
    return dvy_dy + dvx_dx


def gradient_2d(scalar, dx):
    dscalar_dy = np.gradient(scalar, -dx, axis=0)
    dscalar_dx = np.gradient(scalar, dx, axis=1)
    return dscalar_dy, dscalar_dx


def laplacian_2d(scalar, dx):
    d2_dy2 = np.gradient(np.gradient(scalar, -dx, axis=0), -dx, axis=0)
    d2_dx2 = np.gradient(np.gradient(scalar, dx, axis=1), dx, axis=1)
    return d2_dy2 + d2_dx2


def _is_zero(arr):
    return np.allclose(arr, 0.0, atol=1e-14)


def _both_near_zero(computed, expected, atol=1e-8):
    return np.all(np.abs(computed) < atol) and np.all(np.abs(expected) < atol)


def _max_rel_error(computed, expected):
    scale = max(np.max(np.abs(expected)), 1e-10)
    return float(np.max(np.abs(computed - expected)) / scale)


def _median_rel_error(computed, expected):
    scale = max(np.max(np.abs(expected)), 1e-10)
    return float(np.median(np.abs(computed - expected)) / scale)


def check_div_alpha_eq_2kappa(profile, grid, tol):
    dx = grid.pixel_scales[0]

    alpha = profile.deflections_yx_2d_from(grid=grid)
    alpha_y, alpha_x = _to_native_vector(alpha, grid)

    try:
        kappa = profile.convergence_2d_from(grid=grid)
    except (NotImplementedError, TypeError):
        return "SKIP", "convergence not available"

    kappa_2d = _to_native_2d(kappa, grid)

    if _is_zero(kappa_2d):
        return "SKIP", "convergence is zero"

    div_alpha = divergence_2d(alpha_y, alpha_x, dx)

    div_t = trim_border(div_alpha)
    kappa_t = trim_border(kappa_2d)

    if _is_zero(kappa_t):
        return "SKIP", "convergence is zero (interior)"

    if _both_near_zero(div_t, 2.0 * kappa_t):
        return "PASS", "both ~0"

    err = _max_rel_error(div_t, 2.0 * kappa_t)
    med = _median_rel_error(div_t, 2.0 * kappa_t)

    if med <= tol["rtol"]:
        return "PASS", f"med={med:.1e} max={err:.1e}"
    else:
        return "FAIL", f"med={med:.1e} max={err:.1e}"


def check_grad_psi_eq_alpha(profile, grid, tol):
    dx = grid.pixel_scales[0]

    try:
        psi = profile.potential_2d_from(grid=grid)
    except (NotImplementedError, TypeError):
        return "SKIP", "potential not available"

    psi_2d = _to_native_2d(psi, grid)

    if _is_zero(psi_2d):
        return "SKIP", "potential is zero"

    alpha = profile.deflections_yx_2d_from(grid=grid)
    alpha_y, alpha_x = _to_native_vector(alpha, grid)

    grad_y, grad_x = gradient_2d(psi_2d, dx)

    grad_y_t = trim_border(grad_y)
    grad_x_t = trim_border(grad_x)
    alpha_y_t = trim_border(alpha_y)
    alpha_x_t = trim_border(alpha_x)

    if _both_near_zero(grad_y_t, alpha_y_t) and _both_near_zero(grad_x_t, alpha_x_t):
        return "PASS", "both ~0"

    err_y = _max_rel_error(grad_y_t, alpha_y_t)
    err_x = _max_rel_error(grad_x_t, alpha_x_t)
    med_y = _median_rel_error(grad_y_t, alpha_y_t)
    med_x = _median_rel_error(grad_x_t, alpha_x_t)

    med = max(med_y, med_x)
    err = max(err_y, err_x)

    if med <= tol["rtol"]:
        return "PASS", f"med={med:.1e} max={err:.1e}"
    else:
        return "FAIL", f"med={med:.1e} max={err:.1e}"


def check_laplacian_psi_eq_2kappa(profile, grid, tol):
    dx = grid.pixel_scales[0]

    try:
        psi = profile.potential_2d_from(grid=grid)
    except (NotImplementedError, TypeError):
        return "SKIP", "potential not available"

    psi_2d = _to_native_2d(psi, grid)

    if _is_zero(psi_2d):
        return "SKIP", "potential is zero"

    try:
        kappa = profile.convergence_2d_from(grid=grid)
    except (NotImplementedError, TypeError):
        return "SKIP", "convergence not available"

    kappa_2d = _to_native_2d(kappa, grid)

    if _is_zero(kappa_2d):
        return "SKIP", "convergence is zero"

    lap = laplacian_2d(psi_2d, dx)

    lap_t = trim_border(lap)
    kappa_t = trim_border(kappa_2d)

    if _both_near_zero(lap_t, 2.0 * kappa_t):
        return "PASS", "both ~0"

    err = _max_rel_error(lap_t, 2.0 * kappa_t)
    med = _median_rel_error(lap_t, 2.0 * kappa_t)

    if med <= tol["rtol"]:
        return "PASS", f"med={med:.1e} max={err:.1e}"
    else:
        return "FAIL", f"med={med:.1e} max={err:.1e}"


def run_all_checks(name, profile, grid, tol):
    r1_status, r1_detail = check_div_alpha_eq_2kappa(profile, grid, tol)
    r2_status, r2_detail = check_grad_psi_eq_alpha(profile, grid, tol)
    r3_status, r3_detail = check_laplacian_psi_eq_2kappa(profile, grid, tol)

    return {
        "name": name,
        "div_alpha": (r1_status, r1_detail),
        "grad_psi": (r2_status, r2_detail),
        "lap_psi": (r3_status, r3_detail),
    }


def print_summary_table(results_list):
    hdr = f"| {'Profile':<28} | {'div(a)=2k':<16} | {'grad(p)=a':<16} | {'lap(p)=2k':<16} |"
    sep = f"|{'-'*30}|{'-'*18}|{'-'*18}|{'-'*18}|"
    print()
    print(hdr)
    print(sep)

    n_pass = 0
    n_fail = 0
    n_skip = 0

    for r in results_list:
        cols = []
        for key in ("div_alpha", "grad_psi", "lap_psi"):
            status, detail = r[key]
            if status == "PASS":
                cols.append(f"PASS {detail.split(' ')[0]}")
                n_pass += 1
            elif status == "SKIP":
                cols.append(f"SKIP")
                n_skip += 1
            else:
                cols.append(f"FAIL {detail.split(' ')[0]}")
                n_fail += 1

        print(f"| {r['name']:<28} | {cols[0]:<16} | {cols[1]:<16} | {cols[2]:<16} |")

    print()
    print(f"Summary: {n_pass} PASS / {n_fail} FAIL / {n_skip} SKIP")

    if n_fail > 0:
        print()
        print("FAILURES:")
        for r in results_list:
            for key, label in [
                ("div_alpha", "div(a)=2k"),
                ("grad_psi", "grad(p)=a"),
                ("lap_psi", "lap(p)=2k"),
            ]:
                status, detail = r[key]
                if status == "FAIL":
                    print(f"  {r['name']} — {label}: {detail}")

    print()
