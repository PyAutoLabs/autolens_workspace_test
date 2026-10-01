"""
PointSolver Historical Mechanism Replay
=======================================

Validate the finite-difference replay against the actual pre-March source
method, running that method with today's deflection callable. This does not
claim an old full-stack execution. Uses a checked-in, digest-verified extract
from the pinned Git source; installed wheels need no local repository history.

__Contents__
Historical pins; isolated method extraction; numerical replay cross-check.

__Env__
ENV: jax full_datasets
"""

import argparse
import ast
import json
from pathlib import Path
from typing import Tuple

import numpy as np
import autoarray as aa
from error_audit import FIXTURES, tracer_for, fd_magnification
from audit_support import historical_fixture


def measure(evidence):
    fixture = historical_fixture()
    pins = fixture["boundaries"]
    galaxy_sha = fixture["method_source"]["sha"]
    path = fixture["method_source"]["path"]
    source = fixture["source"]
    method = next(
        n
        for n in ast.walk(ast.parse(source))
        if isinstance(n, ast.FunctionDef) and n.name == "hessian_from"
    )
    assert (
        not method.decorator_list
    ), "Review changed historical method before executing"
    namespace = {"aa": aa, "np": np, "Tuple": Tuple}
    exec(compile(ast.Module(body=[method], type_ignores=[]), path, "exec"), namespace)
    comparisons = []
    for row in json.loads(Path(evidence).read_text())["rows"]:
        if row["backend"] != "numpy":
            continue
        tracer = tracer_for(FIXTURES[row["fixture"]])
        points = np.array(row["positions"])
        for step in (row["scale"], 0.01):
            yy, xy, yx, xx = namespace["hessian_from"](
                tracer, points, buffer=step, xp=np
            )
            historical = np.asarray(1 / ((1 - yy) * (1 - xx) - xy * yx))
            replay = fd_magnification(tracer, points, step)
            error = float(np.max(np.abs(historical - replay)))
            assert np.allclose(historical, replay, rtol=1e-12, atol=1e-12)
            comparisons.append(
                {
                    "fixture": row["fixture"],
                    "scale": row["scale"],
                    "precision": row["precision"],
                    "buffer": step,
                    "max_absolute_difference": error,
                }
            )
    if not comparisons:
        raise ValueError("Historical replay requires nonempty NumPy evidence")
    return {
        "boundaries": pins,
        "method_source": {"sha": galaxy_sha, "path": path},
        "evidence_type": "Historical method on current deflections; not an old full-stack run",
        "comparisons": comparisons,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evidence", default="scripts/point_source/solver/current_evidence.json"
    )
    parser.add_argument("--output", default="point_solver_history.json")
    args = parser.parse_args()
    result = measure(args.evidence)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(f"Historical method replay agrees in {len(result['comparisons'])} cases")
