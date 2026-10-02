"""Deterministic synthetic source recovery, independent of the fitted basis.

An affine image-to-source ray map produces a nonuniform source-plane sampling.
Analytic source brightness supplies the data directly, never A @ source. The
least-squares reconstruction is judged at independently inverted mesh nodes,
never by forwarding the solved coefficients through the same candidate basis.
A local #490-style weight mirroring must degrade physical source recovery while
an unchanged uniform control stays bit-identical. No files or sampler needed.
"""

import json
from types import SimpleNamespace
from unittest.mock import patch

import autoarray as aa
from autoarray.inversion.mesh.interpolator import rectangular as adaptive

import numpy as np
from scipy.optimize import brentq
from scipy.special import ndtr

from autoarray.inversion.mesh.interpolator.rectangular import (
    adaptive_rectangular_mappings_weights_via_interpolation_from,
)
from autoarray.inversion.mesh.interpolator.rectangular_uniform import (
    InterpolatorRectangularUniform,
)


class Grid:
    def __init__(self, values):
        self.array = np.asarray(values)
        self.over_sampled = self


def truth(q):
    y, x = q.T
    return np.exp(-((y - 0.18) ** 2 / 0.15 + (x + 0.23) ** 2 / 0.09)) + 0.45 * np.exp(
        -((y + 0.36) ** 2 / 0.06 + (x - 0.31) ** 2 / 0.12)
    )


def cdf_nodes(data, n):
    """Invert the defining Gaussian-mixture CDF by root solving, no knot table."""
    index = np.arange(n * n)
    unit = (np.column_stack([n - index // n, index % n]) - 1) / (n - 3)
    nodes = np.zeros((n * n, 2))
    for d in (0, 1):
        lo, hi = data[:, d].min(), data[:, d].max()
        h = (hi - lo) / n
        raw = lambda z: ndtr((z - data[:, d]) / h).mean()
        a, b = raw(lo), raw(hi)
        for u in np.unique(unit[:, d]):
            if 0 <= u <= 1:
                value = (
                    lo
                    if u == 0
                    else (
                        hi
                        if u == 1
                        else brentq(
                            lambda z: (raw(z) - a) / (b - a) - u, lo, hi, xtol=1e-13
                        )
                    )
                )
                nodes[unit[:, d] == u, d] = value
    return nodes


def solve(mappings, weights, data):
    matrix = np.zeros((len(data), int(mappings.max()) + 1))
    np.add.at(matrix, (np.arange(len(data))[:, None], mappings.clip(min=0)), weights)
    live = np.any(matrix != 0, axis=0)
    source = np.linalg.lstsq(matrix[:, live], data, rcond=None)[0]
    return np.flatnonzero(live), source


def metrics(source, expected):
    return {
        "pearson": float(np.corrcoef(source, expected)[0, 1]),
        "normalized_rms": float(
            np.sqrt(np.mean((source - expected) ** 2)) / np.sqrt(np.mean(expected**2))
        ),
    }


def run():
    n = 16
    axis = np.linspace(-1, 1, 51)
    y, x = np.meshgrid(axis, axis, indexing="ij")
    image = np.column_stack([y.ravel(), x.ravel()])
    source_query = image @ np.array([[0.8, 0.13], [-0.08, 0.9]])
    data = truth(source_query)
    mesh_axis = np.linspace(source_query.min(), source_query.max(), n)
    yy, xx = np.meshgrid(mesh_axis, mesh_axis, indexing="ij")
    nodes = np.column_stack([yy.ravel(), xx.ravel()])

    def uniform_tables():
        interp = InterpolatorRectangularUniform(
            mesh=SimpleNamespace(shape=(n, n)),
            mesh_grid=Grid(nodes),
            data_grid=Grid(source_query),
        )
        mm, _, ww = interp._mappings_sizes_weights
        return mm, ww

    def control():
        return solve(*uniform_tables(), data)[1]

    original = adaptive.adaptive_rectangular_mappings_weights_via_interpolation_from
    m, w = original(
        source_grid_size=n,
        data_grid=source_query,
        data_grid_over_sampled=source_query,
        xp=np,
    )
    physical = cdf_nodes(source_query, n)
    live, source = solve(m, w, data)
    good = metrics(source, truth(physical[live]))
    control_before = control()

    def mirror(*args, **kwargs):
        mm, ww = original(*args, **kwargs)
        return mm, ww[:, [2, 3, 0, 1]]

    # Bracket an actual production entry-point mutation with control runs.
    with patch.object(
        adaptive, "adaptive_rectangular_mappings_weights_via_interpolation_from", mirror
    ):
        bad_m, bad_w = (
            adaptive.adaptive_rectangular_mappings_weights_via_interpolation_from(
                source_grid_size=n,
                data_grid=source_query,
                data_grid_over_sampled=source_query,
                xp=np,
            )
        )
        bad_live, mirrored = solve(bad_m, bad_w, data)
        bad = metrics(mirrored, truth(physical[bad_live]))
        control_after = control()
    assert control_before.tobytes() == control_after.tobytes()
    assert good["pearson"] > 0.995, good
    assert good["normalized_rms"] < 0.07, good
    assert not (bad["pearson"] > 0.995 and bad["normalized_rms"] < 0.07), bad
    cm, cw = uniform_tables()
    broken_control = solve(cm, cw[:, [2, 3, 0, 1]], data)[1]
    assert control_before.tobytes() != broken_control.tobytes()

    # The other families are compared against the same physical analytic truth.
    # Kernel KNN is a smoothing basis rather than an interpolatory nodal basis:
    # allow 20% coefficient RMS while requiring high shape correlation. Judge
    # data-supported nodes only, excluding weakly constrained outer columns.
    family_results = {}
    for name in (
        "uniform",
        "Delaunay",
        "DelaunayNN",
        "KNearestNeighbor",
        "KNNBarycentric",
    ):
        if name == "uniform":
            mm, ww = uniform_tables()
        else:
            mesh = getattr(aa.mesh, name)(pixels=len(nodes))
            interp = mesh.interpolator_cls(
                mesh=mesh, mesh_grid=Grid(nodes), data_grid=Grid(source_query)
            )
            mm, _, ww = interp._mappings_sizes_weights
            mm, ww = np.asarray(mm), np.asarray(ww)

        def physical_metrics(mapping):
            ids, coefficients = solve(mapping, ww, data)
            supported = (np.abs(nodes[ids]) < 0.65).all(axis=1)
            assert supported.sum() >= 36
            return metrics(coefficients[supported], truth(nodes[ids[supported]]))

        correct = physical_metrics(mm)
        # A consistent geometric basis permutation is invisible to residuals:
        # move mappings four rows but retain weights, and judge physical nodes.
        corrupted = np.where(mm >= 0, (mm + 4 * n) % (n * n), -1)
        wrong = physical_metrics(corrupted)
        assert correct["pearson"] > 0.98 and correct["normalized_rms"] < 0.20, (
            name,
            correct,
        )
        assert not (wrong["pearson"] > 0.98 and wrong["normalized_rms"] < 0.20), (
            name,
            wrong,
        )
        family_results[name] = {"correct": correct, "wrong_node_pairing": wrong}
    print(
        json.dumps(
            {
                "adaptive_kernel": {"correct": good, "mirrored": bad},
                "other_families": family_results,
                "control_bit_identical": True,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    run()
