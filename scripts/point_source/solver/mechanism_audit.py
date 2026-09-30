"""
PointSolver Capacity and Dtype Audit
===================================

Opt-in synthetic witnesses for capacity truncation and placeholder promotion.
These exercise current triangle code, not a historical full-stack execution.
Run with ``--output mechanisms.json`` from the workspace root.

__Contents__
Capacity witness; padded-selection dtype witness.

__Env__
ENV: jax full_datasets
"""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from autoarray.structures.triangles.array import ArrayTriangles
from autoarray.structures.triangles.shape import Point
from error_audit import FIXTURES, tracer_for
import autolens as al


def measure():
    rows = []
    # 24 distinct nested triangles all contain the origin. This is a constructed
    # capacity witness, not evidence that either astrophysical fixture overflows.
    vertices = np.concatenate(
        [
            np.array([[-1.0, -1.0], [1.0, -1.0], [0.0, 1.0]]) * r
            for r in np.linspace(1.0, 2.0, 24)
        ]
    )
    indices = np.arange(72).reshape(24, 3)
    for cap in (15, 20, 24):
        triangles = ArrayTriangles(
            indices=jnp.asarray(indices),
            vertices=jnp.asarray(vertices),
            max_containing_size=cap,
        )
        actual = int(np.sum(np.asarray(Point(0.0, 0.0).mask(triangles.triangles))))
        for backend, fun in (
            ("eager", lambda: triangles.containing_indices(Point(0.0, 0.0))),
            ("jit", jax.jit(lambda: triangles.containing_indices(Point(0.0, 0.0)))),
        ):
            selected = np.asarray(fun())
            rows.append(
                {
                    "cap": cap,
                    "backend": backend,
                    "actual": actual,
                    "returned": int(np.sum(selected >= 0)),
                    "indices": selected.tolist(),
                }
            )
            assert actual == 24 and np.sum(selected >= 0) == cap
    dtype_rows = []
    for dtype in (jnp.float32, jnp.float64):
        vertices = jnp.array(
            [[1.123456789012345, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=dtype
        )
        triangles = ArrayTriangles(
            indices=jnp.array([[0, 1, 2], [-1, -1, -1]]), vertices=vertices
        )
        for backend in ("eager", "jit"):

            def outputs():
                return (
                    triangles.triangles,
                    triangles.for_indexes(jnp.array([0, -1])).vertices,
                )

            arrays = (jax.jit(outputs) if backend == "jit" else outputs)()
            dtype_rows.append(
                {
                    "input_dtype": str(vertices.dtype),
                    "backend": backend,
                    "output_dtypes": [str(a.dtype) for a in arrays],
                    "first_vertex_x": float(arrays[0][0, 0, 0]),
                }
            )
            assert all(a.dtype == vertices.dtype for a in arrays)
            assert float(arrays[0][0, 0, 0]) == float(vertices[0, 0])
    physical_counts = []
    for name, fixture in FIXTURES.items():
        tracer = tracer_for(fixture)
        for scale in (0.2, 0.05):
            solver = al.PointSolver.for_limits_and_scale(
                y_min=-4.97,
                y_max=5.03,
                x_min=-4.97,
                x_max=5.03,
                scale=scale,
                pixel_scale_precision=0.0001,
                magnification_threshold=0.1,
            )
            counts = [
                len(step.filtered_triangles)
                for step in solver.steps(
                    tracer=tracer, shape=Point(*fixture["source"]), xp=np
                )
            ]
            physical_counts.append(
                {
                    "fixture": name,
                    "scale": scale,
                    "precision": 0.0001,
                    "counts_per_step": counts,
                    "maximum": max(counts),
                }
            )
    return {
        "physical_capacity_counts": physical_counts,
        "x64": jax.config.x64_enabled,
        "capacity": rows,
        "dtype": dtype_rows,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="point_solver_mechanisms.json")
    args = parser.parse_args()
    if not jax.config.x64_enabled:
        raise RuntimeError("Enable x64 to test both input dtypes")
    result = measure()
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print("Capacity and dtype witnesses passed")
