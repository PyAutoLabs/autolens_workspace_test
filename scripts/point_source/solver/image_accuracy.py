"""
PointSolver Image Accuracy
=========================

Opt-in CPU fp64 research for cluster arc phase 1d. Uncapped NumPy solutions
are compared with independent fixture-specific roots. Conditioning and bounded
polishing are diagnostics, not production error certificates or identity rules.

Run from the workspace root; --summarize validates saved evidence read-only.

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
from scipy.optimize import linear_sum_assignment, root

import autolens as al
from autoarray.structures.triangles.shape import Point
from duplicate_policy import cases_for, physical, reference, solver_for
from error_audit import FIXTURES, match_error, provenance, residual, tracer_for

HERE = Path(__file__).resolve().parent
MAX_TRIANGLES = 200000
PRECISIONS = (1e-3, 1e-4)
CASES = {
    "analytic_quad": ("boundary",),
    "near_caustic": ("cusp_0.001", "cusp_1e-05", "cusp_1e-07"),
}


def hashes():
    return {
        name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
        for name in ("image_accuracy.py", "duplicate_policy.py", "error_audit.py")
    }


def jacobians(tracer, points):
    """LensCalc returns (x,y) components; vectors here are ordered (y,x)."""
    if not len(points):
        return np.empty((0, 2, 2))
    calc = al.LensCalc.from_tracer(tracer=tracer, use_multi_plane=True)
    xy = np.asarray(calc.jacobian_from(grid=jnp.asarray(points), xp=jnp))
    return xy.transpose(2, 0, 1)[:, ::-1, ::-1]


def metrics(tracer, beta, points, refs):
    """Keep zero candidates explicit; never infer completeness from count."""
    if not len(points):
        return {"candidates": [], "coverage": [None] * len(refs)}
    distances = np.linalg.norm(points[:, None] - refs[None], axis=-1)
    matrices = jacobians(tracer, points)
    residuals = residual(tracer, points, beta)
    candidates = []
    for p, d, matrix, r in zip(points, distances, matrices, residuals):
        singular = np.linalg.svd(matrix, compute_uv=False)
        regular = bool(singular[-1] > 1e-14 and np.isfinite(matrix).all())
        correction = np.linalg.solve(matrix, r) if regular else None
        candidates.append(
            {
                "position": p.tolist(),
                "reference_label": int(np.argmin(d)),
                "true_error": float(d.min()),
                "residual": float(np.linalg.norm(r)),
                "singular_values": singular.tolist(),
                "parity": int(np.sign(np.linalg.det(matrix))),
                "linearized_correction": (
                    None if correction is None else float(np.linalg.norm(correction))
                ),
                "status": "regular" if regular else "singular_or_nonfinite",
            }
        )
    return {"candidates": candidates, "coverage": distances.min(axis=0).tolist()}


def trace_steps(solver, tracer, beta, refs):
    steps = []
    triangles = np.empty((0, 3, 2))
    for step in solver.steps(tracer=tracer, shape=Point(*beta), xp=np):
        triangles = np.asarray(step.filtered_triangles.triangles)
        initial_count = len(step.initial_triangles.triangles)
        if initial_count > MAX_TRIANGLES or len(triangles) > MAX_TRIANGLES:
            return steps, triangles, "resource_limited"
        if len(triangles):
            means = triangles.mean(axis=1)
            edge = np.linalg.norm(
                triangles - np.roll(triangles, 1, axis=1), axis=-1
            ).max(axis=1)
            coverage = (
                np.linalg.norm(refs[:, None] - means[None], axis=-1)
                .min(axis=1)
                .tolist()
            )
            # Exact floating point straight-triangle membership, not a certified
            # curved-image or root-uniqueness test; retain centroid distances too.
            inside = [bool(Point(*r).mask(triangles).any()) for r in refs]
            max_edge = float(edge.max())
        else:
            coverage, inside, max_edge = [None] * len(refs), [False] * len(refs), None
        steps.append(
            {
                "step": int(step.number),
                "initial_count": initial_count,
                "kept_count": len(triangles),
                "max_edge": max_edge,
                "reference_in_kept_triangle": inside,
                "coverage": coverage,
            }
        )
        if len(step.up_sampled.triangles) > MAX_TRIANGLES:
            return steps, triangles, "resource_limited"
    return steps, triangles, "complete"


def polish(tracer, beta, points, refs, precision):
    fitted, details = [], []
    for p in points:
        result = root(
            lambda q: residual(tracer, q[None], beta)[0],
            p,
            tol=1e-11,
            options={"maxfev": 100},
        )
        finite = bool(np.isfinite(result.x).all())
        fitted.append(result.x if finite else p)
        details.append(
            {
                "success": bool(result.success),
                "nfev": int(result.nfev),
                "finite": finite,
                "polish_status": int(result.status),
            }
        )
    fitted = np.asarray(fitted).reshape(-1, 2)
    measured = metrics(tracer, beta, fitted, refs)
    for p, original, candidate, detail in zip(
        fitted, points, measured["candidates"], details
    ):
        candidate.update(detail)
        candidate["movement"] = float(np.linalg.norm(p - original))
        candidate["residual_accept"] = (
            detail["finite"] and candidate["residual"] < 1e-10
        )
        correction = candidate["linearized_correction"]
        # A proposed diagnostic screen, deliberately evaluated against the oracle.
        # No root merging, uniqueness claim or guaranteed error bound.
        candidate["condition_accept"] = bool(
            candidate["residual_accept"]
            and correction is not None
            and correction <= precision
        )
    return measured


def run(output):
    if os.environ.get("PYAUTO_SMALL_DATASETS") == "1" or not jax.config.x64_enabled:
        raise RuntimeError("Requires full datasets and JAX x64")
    assert all(d.platform == "cpu" for d in jax.devices()), "CPU-only evidence"
    data = {
        "schema": 1,
        "hashes": hashes(),
        "provenance": provenance(),
        "references": {},
        "rows": [],
        "limits": {
            "extra_levels": 3,
            "max_triangles": MAX_TRIANGLES,
            "polish_maxfev": 100,
        },
    }
    for name, fixture in FIXTURES.items():
        tracer = tracer_for(fixture)
        for case, beta in cases_for(name, fixture, tracer):
            if case not in CASES[name]:
                continue
            ref = reference(tracer, fixture, beta)
            refs = np.asarray(ref["positions"])
            ref["source"] = beta.tolist()
            # Cross-check matrix order and JAX derivative against a centred
            # finite difference of the actual NumPy residual at reference roots.
            h = 1e-5
            fd = np.stack(
                [
                    (
                        residual(tracer, refs + np.eye(2)[i] * h, beta)
                        - residual(tracer, refs - np.eye(2)[i] * h, beta)
                    )
                    / (2 * h)
                    for i in range(2)
                ],
                axis=-1,
            )
            ref["jacobian_fd_max_error"] = float(
                np.max(np.abs(fd - jacobians(tracer, refs)))
            )
            assert ref["jacobian_fd_max_error"] < 1e-7
            data["references"][name + "/" + case] = ref
            for precision in PRECISIONS:
                for extra in range(4):
                    start = time.monotonic()
                    target = precision / 2**extra
                    solver = solver_for(0.2, target)
                    steps, triangles, status = trace_steps(solver, tracer, beta, refs)
                    row = {
                        "fixture": name,
                        "case": case,
                        "precision": precision,
                        "extra": extra,
                        "target_precision": target,
                        "scale": 0.2,
                        "steps": steps,
                        "status": status,
                        "selection_truncated": False,
                    }
                    if status == "complete":
                        points = physical(
                            solver.solve(
                                tracer=tracer,
                                source_plane_coordinate=beta,
                                xp=np,
                                remove_infinities=False,
                            ).array
                        )
                        row["raw"] = metrics(tracer, beta, points, refs)
                        row["polished"] = polish(tracer, beta, points, refs, target)
                        # Production removes the low-magnification central
                        # candidate. Match against identically filtered geometry,
                        # while retaining unfiltered per-step coverage evidence.
                        means = triangles.mean(axis=1)
                        filtered = np.asarray(
                            solver._filter_low_magnification(
                                tracer=tracer, points=means, xp=np
                            )
                        )
                        keep = np.isfinite(filtered).all(axis=1)
                        row["terminal_filtered_count"] = int(np.sum(~keep))
                        triangles, means = triangles[keep], means[keep]
                        error = (
                            0.0
                            if len(points) == len(means) == 0
                            else match_error(points, means)
                        )
                        row["geometry_match_error"] = error
                        row["geometry_verified"] = error is not None and error < 1e-10
                        if row["geometry_verified"] and len(points):
                            a, b = linear_sum_assignment(
                                np.linalg.norm(points[:, None] - means[None], axis=-1)
                            )
                            edges = np.linalg.norm(
                                triangles - np.roll(triangles, 1, axis=1), axis=-1
                            ).max(axis=1)
                            for i, j in zip(a, b):
                                row["raw"]["candidates"][i]["triangle_edge"] = float(
                                    edges[j]
                                )
                        # Step diagnostics remain raw evidence if correspondence
                        # fails; only verified rows receive geometry interpretation.
                        row["first_geometry_absence"] = (
                            [
                                next(
                                    (
                                        s["step"]
                                        for s in steps
                                        if not s["reference_in_kept_triangle"][i]
                                    ),
                                    None,
                                )
                                for i in range(4)
                            ]
                            if row["geometry_verified"]
                            else None
                        )
                    row["seconds"] = time.monotonic() - start
                    data["rows"].append(row)
                    output.write_text(
                        json.dumps(data, indent=2, allow_nan=False) + "\n"
                    )
                    print(
                        name,
                        case,
                        precision,
                        extra,
                        status,
                        len(row.get("raw", {}).get("candidates", [])),
                        f'{row["seconds"]:.2f}s',
                        flush=True,
                    )
    assert data["provenance"] == provenance(), "Library/environment changed during run"
    assert data["hashes"] == hashes(), "Script/helper changed during run"
    data["provenance_verified"] = True
    data["summary"] = summarize(data)
    output.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    return data


def summarize(data):
    assert data["schema"] == 1 and data["provenance_verified"]
    assert data["hashes"] == hashes(), "Evidence script/helper revision mismatch"
    expected = {
        (name, case, p, extra)
        for name, cases in CASES.items()
        for case in cases
        for p in PRECISIONS
        for extra in range(4)
    }
    rows = data["rows"]
    assert len(rows) == len(expected) == 32
    assert {
        (r["fixture"], r["case"], r["precision"], r["extra"]) for r in rows
    } == expected
    b, g = (
        FIXTURES["analytic_quad"]["einstein_radius"],
        FIXTURES["analytic_quad"]["shear"],
    )
    analytic = np.array(
        [[b / (1 + g), 0], [-b / (1 + g), 0], [0, b / (1 - g)], [0, -b / (1 - g)]]
    )
    assert (
        match_error(
            np.asarray(data["references"]["analytic_quad/boundary"]["positions"]),
            analytic,
        )
        < 1e-10
    )
    summary = []
    for row in rows:
        assert row["status"] in ("complete", "resource_limited")
        assert not row["selection_truncated"]
        item = {k: row[k] for k in ("fixture", "case", "precision", "extra", "status")}
        if row["status"] == "complete":
            assert len(row["steps"]) == int(
                np.ceil(np.log2(row["scale"] / row["target_precision"]))
            )
            for key in ("raw", "polished"):
                measured = row[key]
                assert len(measured["coverage"]) == 4
                assert len(measured["candidates"]) == len(row["raw"]["candidates"])
                for c in measured["candidates"]:
                    assert c["status"] in ("regular", "singular_or_nonfinite")
                    if key == "polished":
                        assert isinstance(c["polish_status"], int)
                    assert len(c["singular_values"]) == 2 and c["true_error"] >= 0
                    assert (
                        c["linearized_correction"] is None
                        or c["linearized_correction"] >= 0
                    )
            raw, polished = row["raw"]["candidates"], row["polished"]["candidates"]
            tol = row["target_precision"]
            # Coverage of accepted candidates is computed per independent root;
            # four representatives or a tiny residual alone never suffice.
            accepted = [c for c in polished if c["condition_accept"]]
            refs = np.asarray(
                data["references"][row["fixture"] + "/" + row["case"]]["positions"]
            )
            coverage = (
                np.linalg.norm(
                    refs[:, None] - np.asarray([c["position"] for c in accepted])[None],
                    axis=-1,
                )
                .min(axis=1)
                .tolist()
                if accepted
                else [None] * 4
            )
            separation = data["references"][row["fixture"] + "/" + row["case"]][
                "minimum_separation"
            ]
            if accepted:
                distances = np.linalg.norm(
                    refs[:, None] - np.asarray([c["position"] for c in accepted])[None],
                    axis=-1,
                )
                ri, ci = linear_sum_assignment(distances)
                injective_coverage = bool(
                    len(ri) == 4 and np.all(distances[ri, ci] <= tol)
                )
            else:
                injective_coverage = False
            item.update(
                multiplicity=(
                    "reference_separated"
                    if separation > 2 * tol
                    else "unresolved_at_target"
                ),
                condition_injective_coverage=injective_coverage,
                count=len(raw),
                raw_max_error=max((c["true_error"] for c in raw), default=None),
                raw_covered=sum(
                    d is not None and d <= tol for d in row["raw"]["coverage"]
                ),
                condition_covered=sum(d is not None and d <= tol for d in coverage),
                condition_accepted=len(accepted),
                residual_false_accepts=sum(
                    c["residual_accept"] and c["true_error"] > tol for c in polished
                ),
                condition_false_accepts=sum(
                    c["condition_accept"] and c["true_error"] > tol for c in polished
                ),
                correction_underestimates=sum(
                    c["linearized_correction"] is not None
                    and c["true_error"] > max(2 * c["linearized_correction"], tol)
                    for c in raw
                ),
                geometry_verified=row["geometry_verified"],
            )
        summary.append(item)
    # Enforce strict JSON; all exceptional diagnostics are null + explicit status.
    json.dumps(data, allow_nan=False)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=HERE / "image_accuracy_evidence.json"
    )
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    if args.summarize:
        before = args.output.read_bytes()
        data = json.loads(before)
        assert summarize(data) == data["summary"]
        assert args.output.read_bytes() == before, "Summary changed evidence"
    else:
        data = run(args.output)
    print(json.dumps(data["summary"], indent=2))
