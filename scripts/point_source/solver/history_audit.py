"""
PointSolver Historical Mechanism Replay
=======================================

Validate the finite-difference replay against the actual pre-March source
method, running that method with today's deflection callable. This does not
claim an old full-stack execution. Requires local git history (read only).

__Contents__
Historical pins; isolated method extraction; numerical replay cross-check.

__Env__
ENV: jax full_datasets
"""

import argparse
import ast
import json
from pathlib import Path
import subprocess
from typing import Tuple

import numpy as np
import autoarray as aa
import autogalaxy as ag
import autolens as al
from error_audit import FIXTURES, tracer_for, fd_magnification


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True)


def measure(evidence):
    lens = Path(al.__file__).resolve().parents[1]
    array = Path(aa.__file__).resolve().parents[1]
    galaxy = Path(ag.__file__).resolve().parents[1]
    boundaries = [
        (array, "e0e2f28e"),
        (array, "314e2d09"),
        (lens, "0ea7c6000"),
        (lens, "c36f8a6ec"),
        (lens, "dd82ce386"),
        (lens, "14826f6c7"),
        (lens, "3db51dd38"),
        (galaxy, "ee92bebe"),
        (galaxy, "eacdcd77"),
        (lens, "5c42d8133"),
        (lens, "fca58c468"),
        (lens, "d24339c37"),
    ]
    pins = [
        {
            "repo": p.name,
            "sha": git(p, "rev-parse", sha).strip(),
            "parent": git(p, "rev-parse", sha + "^").strip(),
            "date_subject": git(p, "show", "-s", "--format=%cI %s", sha).strip(),
        }
        for p, sha in boundaries
    ]
    stamp = git(lens, "show", "-s", "--format=%cI", "14826f6c7^").strip()
    galaxy_sha = git(galaxy, "rev-list", "-1", "--before=" + stamp, "main").strip()
    path = "autogalaxy/operate/deflections.py"
    source = git(galaxy, "show", galaxy_sha + ":" + path)
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
