"""
PointSolver Duplicate Policy
===========================

Opt-in, full-data CPU research for cluster arc phase 1c. Run from the workspace
root with ``python scripts/point_source/solver/duplicate_policy.py``. Candidate
policies are host-side diagnostics, never changes to the production solver.

__Contents__
Independent angular roots; boundary and cusp controls; triangle diagnostics;
candidate policies; backend and vmap evidence.

__Env__
ENV: jax full_datasets
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import brentq, root

import autolens as al
from autoarray.structures.triangles.shape import Point
from error_audit import (
    FIXTURES,
    deflections,
    match_error,
    provenance,
    residual,
    tracer_for,
)


"""__Independent References__

Both existing fixtures have an isothermal mass component, whose deflection is
independent of radius, and optionally linear external shear. Eliminate radius
from beta = r (u - shear(u)) - alpha_mass(u), then bracket the angular roots.
This uses no solver centroids or candidate grouping. Two angular grids must
agree, and the reconstructed positions must satisfy the full lens equation.
It is a fixture-specific oracle, not a general lens solver.
"""


def angular_roots(tracer, fixture, beta, n):
    shear = fixture.get("shear", 0.0)

    def equation(angles):
        u = np.column_stack((np.sin(angles), np.cos(angles)))
        linear = u * np.array([-shear, shear])
        v = u - linear
        w = np.asarray(beta) + deflections(tracer, u) - linear
        return v[:, 0] * w[:, 1] - v[:, 1] * w[:, 0]

    # Dense plus logarithmically spaced angles resolve the cusp neighbourhood
    # without depending on the triangle solution. All four diagonals included.
    fine = np.geomspace(1e-7, 0.1, 250)
    special = np.concatenate(
        [a + np.r_[-fine, 0.0, fine] for a in np.arange(4) * np.pi / 2 + np.pi / 4]
    )
    angles = np.unique(np.r_[np.linspace(0, 2 * np.pi, n + 1), special])
    values = equation(angles)
    solutions = []
    for i in range(len(angles) - 1):
        if values[i] == 0:
            solutions.append(angles[i])
        elif values[i] * values[i + 1] < 0:
            solutions.append(
                brentq(
                    lambda a: equation(np.array([a]))[0],
                    angles[i],
                    angles[i + 1],
                    xtol=5e-15,
                )
            )
    positions = []
    for a in solutions:
        u = np.array([np.sin(a), np.cos(a)])
        linear = u * np.array([-shear, shear])
        v = u - linear
        w = beta + deflections(tracer, u[None])[0] - linear
        radius = np.dot(v, w) / np.dot(v, v)
        p = radius * u
        if radius > 0 and all(np.linalg.norm(p - q) > 1e-6 for q in positions):
            positions.append(p)
    return np.array(sorted(positions, key=lambda p: tuple(p)))


def reference(tracer, fixture, beta):
    coarse = angular_roots(tracer, fixture, beta, 4096)
    fine = angular_roots(tracer, fixture, beta, 8192)
    error = match_error(coarse, fine)
    assert len(fine) == 4 and error is not None and error < 1e-6, (beta, coarse, fine)
    max_residual = float(np.linalg.norm(residual(tracer, fine, beta), axis=1).max())
    assert max_residual < 1e-10
    d = np.linalg.norm(fine[:, None] - fine[None, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    return {
        "positions": fine.tolist(),
        "grid_convergence_error": error,
        "residual_max": max_residual,
        "minimum_separation": float(d.min()),
    }


def cases_for(name, fixture, tracer):
    if name == "analytic_quad":
        return [
            ("boundary", np.array([0.0, 0.0])),
            ("offset_plus", np.array([1e-5, 2e-5])),
            ("offset_minus", np.array([-1e-5, -2e-5])),
        ]
    # Locate the symmetry-axis cusp independently via the critical eigenvalue.
    u = np.ones(2) / np.sqrt(2)
    calc = al.LensCalc.from_tracer(tracer=tracer, use_multi_plane=True)
    critical_r = brentq(
        lambda r: 1.0
        / float(
            calc.magnification_2d_via_hessian_from(grid=jnp.asarray([r * u]), xp=jnp)[0]
        ),
        1.3,
        1.9,
    )
    cusp = residual(tracer, np.array([critical_r * u]), [0, 0])[0]
    return [
        (f"cusp_{fraction:g}", cusp * (1 - fraction)) for fraction in (1e-3, 1e-5, 1e-7)
    ]


"""__Solver Measurements__

Call the production solve separately from the instrumented steps: returning
additional intermediates can change XLA optimisation at an exact boundary.
The latter records uncapped counts and final triangles without monkeypatches.
PyAutoLens:autolens/point/solver/{point_solver,shape_solver}.py.
"""


def solver_for(scale, precision):
    return al.PointSolver.for_limits_and_scale(
        y_min=-4.97,
        y_max=5.03,
        x_min=-4.97,
        x_max=5.03,
        scale=scale,
        pixel_scale_precision=precision,
        magnification_threshold=0.1,
    )


def measure_steps(solver, tracer, beta, xp):
    counts = []
    for step in solver.steps(tracer=tracer, shape=Point(*beta), xp=xp):
        plane = step.plane_triangles
        # Match ArrayTriangles' step-0 containment dispatch, before capped where.
        mask = None
        if getattr(plane, "step0_layout", None) is not None:
            mask = plane._step0_point_mask(Point(*beta))
        if mask is None:
            mask = Point(*beta).mask(plane.triangles)
        counts.append(xp.sum(mask))
    return xp.stack(counts), step.filtered_triangles.triangles


def physical(array):
    array = np.asarray(array)
    return array[np.isfinite(array).all(axis=1)]


def parity(tracer, points):
    if not len(points):
        return np.empty(0)
    calc = al.LensCalc.from_tracer(tracer=tracer, use_multi_plane=True)
    mu = np.asarray(
        calc.magnification_2d_via_hessian_from(grid=jnp.asarray(points), xp=jnp)
    )
    return np.sign(mu)


"""__Candidate Policies__

Distance is a negative control. Shared-edge grouping tests whether topology
alone suffices. Root polishing tests identity with a much tighter tolerance
independent of requested triangle precision, with parity and a displacement
bound. None of these is treated as the oracle; all are checked against angular
roots, and missing input images cannot be repaired by deduplication.
"""


def greedy_groups(points, predicate):
    # Lexicographic order makes representative selection independent of row order.
    groups = []
    for i in sorted(range(len(points)), key=lambda k: tuple(points[k])):
        group = next((g for g in groups if predicate(i, g[0])), None)
        if group is None:
            groups.append([i])
        else:
            group.append(i)
    return groups


def evaluate(points, groups, refs, precision):
    representatives = np.asarray([points[g[0]] for g in groups]).reshape(-1, 2)
    error = match_error(representatives, refs)
    labels = np.argmin(np.linalg.norm(points[:, None] - refs[None, :], axis=-1), axis=1)
    return {
        "groups": groups,
        "representatives": representatives.tolist(),
        "count": len(groups),
        "position_error_max": error,
        "four_within_precision": error is not None and error <= 2 * precision,
        "merged_distinct_reference_labels": any(
            len(set(labels[g])) > 1 for g in groups
        ),
    }


def policies(tracer, beta, points, triangles, refs, precision):
    result = {}
    if not len(points):
        return {"empty_input": True}
    distance = lambda i, j: np.linalg.norm(points[i] - points[j]) <= 2 * precision
    result["distance"] = evaluate(
        points, greedy_groups(points, distance), refs, precision
    )
    # Associate each raw output with the independently instrumented centroid.
    # Refuse geometry interpretation if instrumentation changed physical output.
    means = triangles.mean(axis=1)
    nearest = np.argmin(
        np.linalg.norm(points[:, None] - means[None, :], axis=-1), axis=1
    )
    correspondence = np.max(np.linalg.norm(points - means[nearest], axis=1))
    result["triangle_correspondence_error"] = float(correspondence)
    if correspondence < 1e-10:

        def adjacent(i, j):
            a, b = triangles[nearest[i]], triangles[nearest[j]]
            shared = (
                np.min(np.linalg.norm(a[:, None] - b[None, :], axis=-1), axis=1) < 1e-10
            )
            return np.sum(shared) >= 2

        result["shared_edge"] = evaluate(
            points, greedy_groups(points, adjacent), refs, precision
        )
    signs = parity(tracer, points)
    polished, valid, residuals = [], [], []
    for p in points:
        fitted = root(lambda q: residual(tracer, q[None], beta)[0], p, tol=1e-11).x
        r = float(np.linalg.norm(residual(tracer, fitted[None], beta)))
        polished.append(fitted)
        residuals.append(r)
        valid.append(r < 1e-10 and np.linalg.norm(fitted - p) <= 2 * precision)
    polished = np.asarray(polished)
    polished_signs = parity(tracer, polished)
    valid = np.asarray(valid) & (signs == polished_signs)

    def same_root(i, j):
        return (
            valid[i]
            and valid[j]
            and signs[i] == signs[j]
            and np.linalg.norm(polished[i] - polished[j]) < 1e-7
        )

    groups = greedy_groups(points, same_root)
    result["root_identity"] = evaluate(points, groups, refs, precision)
    result["root_identity"].update(
        polished_positions=polished.tolist(),
        polished_residuals=residuals,
        valid=valid.tolist(),
        parity=signs.tolist(),
        polished_nearest_reference_distances=np.min(
            np.linalg.norm(polished[:, None] - refs[None, :], axis=-1), axis=1
        ).tolist(),
    )
    # Changing input order must not change selected physical representatives.
    reversed_points = points[::-1]
    reverse_groups = greedy_groups(
        reversed_points,
        lambda i, j: same_root(len(points) - 1 - i, len(points) - 1 - j),
    )
    reverse_reps = np.array([reversed_points[g[0]] for g in reverse_groups])
    assert (
        match_error(
            reverse_reps, np.asarray(result["root_identity"]["representatives"])
        )
        == 0
    )
    return result


def run(args):
    if os.environ.get("PYAUTO_SMALL_DATASETS") == "1" or not jax.config.x64_enabled:
        raise RuntimeError("Requires full datasets and JAX x64")
    script_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result = {
        "provenance": provenance(),
        "references": {},
        "rows": [],
        "vmap": [],
        "script_sha256": script_hash,
        "fixture_helper_sha256": hashlib.sha256(
            Path(__file__).with_name("error_audit.py").read_bytes()
        ).hexdigest(),
    }

    def save():
        Path(args.output).write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n"
        )

    for name, fixture in FIXTURES.items():
        tracer = tracer_for(fixture)
        cases = cases_for(name, fixture, tracer)
        refs = {}
        for case, beta in cases:
            refs[case] = reference(tracer, fixture, beta)
            refs[case]["source"] = beta.tolist()
            print("reference", name, case, refs[case]["minimum_separation"], flush=True)
        result["references"][name] = refs
        scales = (0.2, 0.05) if name == "analytic_quad" else (0.2,)
        for scale in scales:
            for precision in (1e-3, 1e-4):
                solver = solver_for(scale, precision)
                for backend, xp in (("numpy", np), ("jax", jnp), ("jit", jnp)):

                    def solve(beta):
                        return solver.solve(
                            tracer=tracer,
                            source_plane_coordinate=beta,
                            xp=xp,
                            remove_infinities=False,
                        ).array

                    def diagnostics(beta):
                        return measure_steps(solver, tracer, beta, xp)

                    solve_fn = jax.jit(solve) if backend == "jit" else solve
                    diagnostic_fn = (
                        jax.jit(diagnostics) if backend == "jit" else diagnostics
                    )
                    for case, beta in cases:
                        start = time.monotonic()
                        array = np.asarray(solve_fn(xp.asarray(beta)))
                        points = physical(array)
                        counts, triangles = diagnostic_fn(xp.asarray(beta))
                        counts, triangles = np.asarray(counts), np.asarray(triangles)
                        triangles = triangles[np.isfinite(triangles).all(axis=(1, 2))]
                        reference_positions = np.asarray(refs[case]["positions"])
                        exceeds_cap = bool(
                            np.max(counts) > result["provenance"]["max_containing_size"]
                        )
                        row = {
                            "fixture": name,
                            "case": case,
                            "scale": scale,
                            "precision": precision,
                            "backend": backend,
                            "source": beta.tolist(),
                            "shape": list(array.shape),
                            "positions": points.tolist(),
                            "count": len(points),
                            "residuals": np.linalg.norm(
                                residual(tracer, points, beta), axis=1
                            ).tolist(),
                            "parity": parity(tracer, points).tolist(),
                            "counts_per_step": counts.tolist(),
                            "exceeds_jax_capacity": exceeds_cap,
                            "selection_truncated": backend != "numpy" and exceeds_cap,
                            "reference_coverage": {
                                "nearest_candidate_distances": np.min(
                                    np.linalg.norm(
                                        reference_positions[:, None] - points[None, :],
                                        axis=-1,
                                    ),
                                    axis=1,
                                ).tolist(),
                                "candidate_nearest_reference_distances": np.min(
                                    np.linalg.norm(
                                        points[:, None] - reference_positions[None, :],
                                        axis=-1,
                                    ),
                                    axis=1,
                                ).tolist(),
                            },
                            "triangles": triangles.tolist(),
                            "reference_pair_resolved": refs[case]["minimum_separation"]
                            > 2 * precision,
                            "policies": policies(
                                tracer,
                                beta,
                                points,
                                triangles,
                                reference_positions,
                                precision,
                            ),
                            "seconds_including_compile": time.monotonic() - start,
                        }
                        result["rows"].append(row)
                        save()
                        print(
                            name,
                            case,
                            scale,
                            precision,
                            backend,
                            "count",
                            len(points),
                            "cap",
                            exceeds_cap,
                            flush=True,
                        )
                    if backend == "jit" and precision == 1e-4 and scale == scales[-1]:
                        batched = np.asarray(
                            jax.jit(jax.vmap(solve))(jnp.asarray([b for _, b in cases]))
                        )
                        for (case, beta), array in zip(cases, batched):
                            points = physical(array)
                            scalar = next(
                                r
                                for r in reversed(result["rows"])
                                if r["case"] == case and r["backend"] == "jit"
                            )
                            result["vmap"].append(
                                {
                                    "fixture": name,
                                    "case": case,
                                    "shape": list(array.shape),
                                    "positions": points.tolist(),
                                    "count": len(points),
                                    "scalar_match_error": match_error(
                                        points, np.asarray(scalar["positions"])
                                    ),
                                    "policies": policies(
                                        tracer,
                                        beta,
                                        points,
                                        np.asarray(scalar["triangles"]),
                                        np.asarray(refs[case]["positions"]),
                                        precision,
                                    ),
                                }
                            )
                        save()
                jax.clear_caches()
    assert (
        provenance() == result["provenance"]
    ), "Library/environment changed during audit"
    assert (
        hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == script_hash
    ), "Script changed during audit"
    result["provenance_verified"] = True
    print("PASS: library/environment and script provenance unchanged", flush=True)
    save()
    return result


def summarize(result):
    """Validate the saved matrix and expose counterexamples, without re-solving."""
    rows = result["rows"]
    keys = {
        (r["fixture"], r["case"], r["scale"], r["precision"], r["backend"])
        for r in rows
    }
    assert len(keys) == len(rows) == 54, "Incomplete or duplicated evidence matrix"
    assert len(result["vmap"]) == 6, "Incomplete vmap controls"
    b, g = (
        FIXTURES["analytic_quad"]["einstein_radius"],
        FIXTURES["analytic_quad"]["shear"],
    )
    analytic = np.array(
        [[b / (1 + g), 0], [-b / (1 + g), 0], [0, b / (1 - g)], [0, -b / (1 - g)]]
    )
    assert (
        match_error(
            np.asarray(result["references"]["analytic_quad"]["boundary"]["positions"]),
            analytic,
        )
        < 1e-10
    )
    assert all(
        r["policies"]["root_identity"]["four_within_precision"]
        for r in rows
        if r["fixture"] == "analytic_quad"
    )
    controls = []
    for fixture, cases in result["references"].items():
        for case, ref in cases.items():
            points = np.asarray(ref["positions"])
            for precision in (1e-3, 1e-4):
                groups = greedy_groups(
                    points,
                    lambda i, j: np.linalg.norm(points[i] - points[j]) <= 2 * precision,
                )
                controls.append(
                    {
                        "fixture": fixture,
                        "case": case,
                        "precision": precision,
                        "distance_count_on_true_roots": len(groups),
                    }
                )
    result["reference_distance_controls"] = controls
    result["summary"] = {
        "rows": len(rows),
        "vmap_rows": len(result["vmap"]),
        "raw_wrong_cardinality_rows": sum(r["count"] != 4 for r in rows),
        "exceeds_jax_capacity_rows": sum(r["exceeds_jax_capacity"] for r in rows),
        "selection_truncated_rows": sum(r["selection_truncated"] for r in rows),
        "maximum_uncapped_containment": max(max(r["counts_per_step"]) for r in rows),
        "distance_merges_true_roots_controls": [
            r for r in controls if r["distance_count_on_true_roots"] != 4
        ],
        "policies": {
            policy: {
                "evaluated_rows": sum(policy in r["policies"] for r in rows),
                "untruncated_resolved_rows": sum(
                    r["reference_pair_resolved"] and not r["selection_truncated"]
                    for r in rows
                ),
                "four_within_precision_rows": sum(
                    r["policies"].get(policy, {}).get("four_within_precision", False)
                    for r in rows
                ),
                "resolved_failures": [
                    {
                        k: r[k]
                        for k in (
                            "fixture",
                            "case",
                            "scale",
                            "precision",
                            "backend",
                            "count",
                        )
                    }
                    for r in rows
                    if r["reference_pair_resolved"]
                    and not r["selection_truncated"]
                    and not r["policies"]
                    .get(policy, {})
                    .get("four_within_precision", False)
                ],
            }
            for policy in ("distance", "shared_edge", "root_identity")
        },
        "vmap_scalar_mismatches": sum(
            r["scalar_match_error"] is None or r["scalar_match_error"] > 1e-10
            for r in result["vmap"]
        ),
    }
    return result["summary"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", default="scripts/point_source/solver/duplicate_policy_evidence.json"
    )
    parser.add_argument(
        "--summarize",
        action="store_true",
        help="Validate and summarize an existing complete evidence file",
    )
    args = parser.parse_args()
    if args.summarize:
        result = json.loads(Path(args.output).read_text())
        saved_summary = result["summary"]
        assert result["provenance_verified"], "Missing completed provenance check"
        assert (
            result["script_sha256"]
            == hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        ), "Evidence belongs to a different script revision"
        assert (
            summarize(result) == saved_summary
        ), "Saved summary differs from recomputed evidence"
        print(json.dumps(result["summary"], indent=2))
    else:
        result = run(args)
        print(json.dumps(summarize(result), indent=2))
        Path(args.output).write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n"
        )
