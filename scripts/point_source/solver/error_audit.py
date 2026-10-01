"""
PointSolver Error Audit
======================

Opt-in numerical evidence for cluster arc phase 1a. Run from the workspace root:
``python scripts/point_source/solver/error_audit.py --output audit.json``.
Historical finite-difference rows are mechanism replays on the pinned current
mass model, not executions of old stacks. No library or dataset is modified.

__Contents__
Fixtures; independent references; solve and filter measurements; JSON evidence.

__Env__
ENV: jax full_datasets
"""

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import time

import numpy as np
from scipy.optimize import linear_sum_assignment, root
import jax
import jax.numpy as jnp
import autolens as al
from autoarray.structures.triangles.array import MAX_CONTAINING_SIZE
from audit_support import library_provenance


"""__Fixtures__

The centred SIS plus external shear has four analytic axis images. The SIE
fixture freezes autolens_profiling/dataset/point_source/near_caustic/truth.json
at the provenance SHA in RESULTS.md (q=0.9, angle=45 degrees, theta_E=1.6).
All coordinates, scales and residuals are in arcseconds; magnification is signed.
"""

FIXTURES = {
    "analytic_quad": {
        "source": [0.0, 0.0],
        "einstein_radius": 1.6,
        "shear": 0.1,
        "ell_comps": [0.0, 0.0],
    },
    "near_caustic": {
        "source": [0.07777695929510982, 0.07777695971376672],
        "einstein_radius": 1.6,
        "ell_comps": [0.05263157894736841, 3.2227547345982974e-18],
    },
}


def tracer_for(fixture):
    profiles = {
        "mass": al.mp.Isothermal(
            centre=(0.0, 0.0),
            einstein_radius=fixture["einstein_radius"],
            ell_comps=tuple(fixture["ell_comps"]),
        )
    }
    if "shear" in fixture:
        profiles["shear"] = al.mp.ExternalShear(gamma_1=fixture["shear"], gamma_2=0.0)
    return al.Tracer(
        galaxies=[al.Galaxy(redshift=0.5, **profiles), al.Galaxy(redshift=1.0)]
    )


def deflections(tracer, points):
    return np.asarray(tracer.deflections_yx_2d_from(grid=al.Grid2DIrregular(points)))


def residual(tracer, points, source):
    return points - deflections(tracer, points) - np.asarray(source)


def reference_for(name, fixture, tracer):
    if name == "analytic_quad":
        b, g = fixture["einstein_radius"], fixture["shear"]
        positions = np.array(
            [[b / (1 + g), 0], [-b / (1 + g), 0], [0, b / (1 - g)], [0, -b / (1 - g)]]
        )
    else:
        # Independent nonlinear lens-equation roots from a dense ring of seeds;
        # repeat at tighter tolerance and require the same four roots.
        roots_by_tol = []
        for tol in (1e-9, 1e-11):
            positions = []
            for angle in np.linspace(0, 2 * np.pi, 128, endpoint=False):
                result = root(
                    lambda p: residual(tracer, p[None, :], fixture["source"])[0],
                    1.6 * np.array([np.sin(angle), np.cos(angle)]),
                    tol=tol,
                )
                p = result.x
                if (
                    np.linalg.norm(residual(tracer, p[None, :], fixture["source"]))
                    < 1e-9
                ):
                    if all(np.linalg.norm(p - q) > 1e-5 for q in positions):
                        positions.append(p)
            roots_by_tol.append(np.asarray(positions))
        assert all(
            len(p) == 4 for p in roots_by_tol
        ), "Reference did not recover four roots"
        assert match_error(*roots_by_tol) < 1e-8, "Reference failed convergence check"
        positions = roots_by_tol[-1]
    assert (
        np.max(np.linalg.norm(residual(tracer, positions, fixture["source"]), axis=1))
        < 1e-8
    )
    return positions


def match_error(points, reference):
    if len(points) != len(reference):
        return None
    distance = np.linalg.norm(points[:, None, :] - reference[None, :, :], axis=-1)
    a, b = linear_sum_assignment(distance)
    return float(distance[a, b].max())


"""__Magnification Mechanisms__

Replay the old centred two-point Hessian at the initial triangle scale and at
0.01. The current NumPy adaptive Richardson and JAX jacfwd paths are measured
separately. Filtering changes membership; it does not move a surviving centroid.
"""


def fd_magnification(tracer, points, step):
    dy, dx = np.array([step, 0.0]), np.array([0.0, step])
    yy, xy = (
        (deflections(tracer, points + dy) - deflections(tracer, points - dy))
        / (2 * step)
    ).T
    yx, xx = (
        (deflections(tracer, points + dx) - deflections(tracer, points - dx))
        / (2 * step)
    ).T
    return 1 / ((1 - yy) * (1 - xx) - xy * yx)


def magnifications(tracer, points, scale):
    calc = al.LensCalc.from_tracer(tracer=tracer, use_multi_plane=True)
    return {
        "old_fd_scale": fd_magnification(tracer, points, scale),
        "old_fd_0.01": fd_magnification(tracer, points, 0.01),
        "current_numpy": np.asarray(
            calc.magnification_2d_via_hessian_from(grid=points)
        ),
        "current_jax": np.asarray(
            calc.magnification_2d_via_hessian_from(grid=jnp.asarray(points), xp=jnp)
        ),
    }


def provenance():
    import autoarray, autogalaxy, autofit, autonerves

    repos = {
        module.__name__: library_provenance(module)
        for module in (al, autoarray, autogalaxy, autofit, autonerves)
    }
    return {
        "libraries": repos,
        "packages": {
            p: importlib.metadata.version(p)
            for p in ("numpy", "scipy", "jax", "jaxlib", "numba")
        },
        "x64": jax.config.x64_enabled,
        "max_containing_size": MAX_CONTAINING_SIZE,
        "devices": [str(d) for d in jax.devices()],
    }


def run(args):
    if os.environ.get("PYAUTO_SMALL_DATASETS") == "1":
        raise RuntimeError(
            "Unset PYAUTO_SMALL_DATASETS: this audit requires real solves"
        )
    if not jax.config.x64_enabled:
        raise RuntimeError("Enable JAX x64 before running the audit")
    output = {"provenance": provenance(), "fixtures": FIXTURES, "rows": []}
    if args.resume and Path(args.output).exists():
        previous = json.loads(Path(args.output).read_text())
        assert (
            previous["provenance"] == output["provenance"]
        ), "Cannot resume with different library/environment pins"
        output["rows"] = previous["rows"]
    for name, fixture in FIXTURES.items():
        tracer = tracer_for(fixture)
        reference = reference_for(name, fixture, tracer)
        fixture["reference_positions"] = reference.tolist()
        for scale in args.scales:
            for precision in args.precisions:
                solver = al.PointSolver.for_limits_and_scale(
                    y_min=-4.97,
                    y_max=5.03,
                    x_min=-4.97,
                    x_max=5.03,
                    scale=scale,
                    pixel_scale_precision=precision,
                    magnification_threshold=0.0,
                )
                for backend in args.backends:
                    if any(
                        (r["fixture"], r["scale"], r["precision"], r["backend"])
                        == (name, scale, precision, backend)
                        for r in output["rows"]
                    ):
                        continue
                    xp = np if backend == "numpy" else jnp

                    def solve(beta):
                        return solver.solve(
                            tracer=tracer,
                            source_plane_coordinate=beta,
                            xp=xp,
                            remove_infinities=False,
                        ).array

                    start = time.monotonic()
                    array = np.asarray(
                        (jax.jit(solve) if backend == "jit" else solve)(
                            xp.asarray(fixture["source"])
                        )
                    )
                    elapsed = time.monotonic() - start
                    points = array[np.isfinite(array).all(axis=1)]
                    mu = magnifications(tracer, points, scale)
                    row = {
                        "fixture": name,
                        "scale": scale,
                        "precision": precision,
                        "backend": backend,
                        "shape": list(array.shape),
                        "dtype": str(array.dtype),
                        "n_images": len(points),
                        "positions": points.tolist(),
                        "position_error_max": match_error(points, reference),
                        "residual_max": float(
                            np.max(
                                np.linalg.norm(
                                    residual(tracer, points, fixture["source"]), axis=1
                                )
                            )
                        ),
                        "seconds_including_compile": elapsed,
                        "magnifications": {k: v.tolist() for k, v in mu.items()},
                        "filtered_position_errors": {
                            str(t): {
                                k: match_error(points[np.abs(v) > t], reference)
                                for k, v in mu.items()
                            }
                            for t in (1e-8, 0.1, 5.0, 10.0, 20.0, 50.0)
                        },
                        "filter_counts": {
                            str(t): {
                                k: int(np.sum(np.abs(v) > t)) for k, v in mu.items()
                            }
                            for t in (1e-8, 0.1, 5.0, 10.0, 20.0, 50.0)
                        },
                    }
                    solver.magnification_threshold = 0.1

                    def solve_default(beta):
                        return solver.solve(
                            tracer=tracer, source_plane_coordinate=beta, xp=xp
                        ).array

                    try:
                        default_array = np.asarray(
                            (
                                jax.jit(solve_default)
                                if backend == "jit"
                                else solve_default
                            )(xp.asarray(fixture["source"]))
                        )
                        row["default_exception"] = None
                    except jax.errors.NonConcreteBooleanIndexError as error:
                        # Record the documented xp override's default-shape failure,
                        # then measure the explicitly padded supported route.
                        row["default_exception"] = type(error).__name__

                        def solve_explicit_padded(beta):
                            return solver.solve(
                                tracer=tracer,
                                source_plane_coordinate=beta,
                                xp=xp,
                                remove_infinities=False,
                            ).array

                        default_array = np.asarray(
                            jax.jit(solve_explicit_padded)(
                                xp.asarray(fixture["source"])
                            )
                        )
                    physical = default_array[np.isfinite(default_array).all(axis=1)]
                    row["measured_filtered_shape"] = list(default_array.shape)
                    row["default_shape"] = (
                        list(default_array.shape)
                        if row["default_exception"] is None
                        else None
                    )
                    row["default_positions"] = physical.tolist()
                    row["default_error_max"] = match_error(physical, reference)
                    row["default_has_four_images"] = len(physical) == 4
                    row["default_within_precision"] = (
                        row["default_error_max"] is not None
                        and row["default_error_max"] < 2 * precision
                    )
                    unique = []
                    for point in physical:
                        if all(
                            np.linalg.norm(point - other) > 2 * precision
                            for other in unique
                        ):
                            unique.append(point)
                    row["diagnostic_unique_count"] = len(unique)
                    row["diagnostic_unique_error"] = match_error(
                        np.asarray(unique), reference
                    )
                    assert (
                        match_error(physical, points[np.abs(mu["current_numpy"]) > 0.1])
                        < 1e-8
                    )
                    solver.magnification_threshold = 0.0
                    output["rows"].append(row)
                    Path(args.output).write_text(
                        json.dumps(output, indent=2, allow_nan=False) + "\n"
                    )
                    print(
                        name,
                        scale,
                        precision,
                        backend,
                        len(points),
                        row["default_error_max"],
                        row["default_exception"],
                        flush=True,
                    )
                    jax.clear_caches()
    # Normalize earlier resumed rows and expose cardinality separately from
    # a diagnostic grouping. Never replace the measured solver output.
    for row in output["rows"]:
        points = np.asarray(row["default_positions"])
        reference = np.asarray(
            output["fixtures"][row["fixture"]]["reference_positions"]
        )
        row["measured_filtered_shape"] = row.get(
            "measured_filtered_shape", row["default_shape"]
        )
        if row["default_exception"] is not None:
            row["default_shape"] = None
        row["default_has_four_images"] = len(points) == 4
        row["default_within_precision"] = (
            row["default_error_max"] is not None
            and row["default_error_max"] < 2 * row["precision"]
        )
        unique = []
        for point in points:
            if all(
                np.linalg.norm(point - other) > 2 * row["precision"] for other in unique
            ):
                unique.append(point)
        row["diagnostic_unique_count"] = len(unique)
        row["diagnostic_unique_error"] = match_error(np.asarray(unique), reference)
        candidates = np.asarray(row["positions"])
        row["filter_kept_indices"] = {
            str(t): {
                k: np.flatnonzero(np.abs(v) > t).tolist()
                for k, v in row["magnifications"].items()
            }
            for t in (1e-8, 0.01, 0.1, 5.0, 10.0, 20.0, 50.0, 200.0)
        }
        row["filter_counts"] = {
            t: {k: len(indices) for k, indices in methods.items()}
            for t, methods in row["filter_kept_indices"].items()
        }
        row["filtered_position_errors"] = {
            t: {
                k: match_error(candidates[indices], reference)
                for k, indices in methods.items()
            }
            for t, methods in row["filter_kept_indices"].items()
        }
    assert (
        provenance() == output["provenance"]
    ), "Library/environment changed during audit"
    Path(args.output).write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output", default="point_solver_error_audit.json")
    parser.add_argument("--scales", type=float, nargs="+", default=[0.2, 0.05])
    parser.add_argument("--precisions", type=float, nargs="+", default=[0.001, 0.0001])
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=["numpy", "jax", "jit"],
        default=["numpy", "jax", "jit"],
    )
    run(parser.parse_args())
