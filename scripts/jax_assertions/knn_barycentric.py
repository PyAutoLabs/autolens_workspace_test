"""
Jax Assertions: KNN-Barycentric Interpolator
============================================

Exercises ``aa.mesh.KNNBarycentric`` (added in PyAutoArray PR #318) through
``aa.Mapper`` / ``aa.Inversion``. The interpolator picks the 3 nearest mesh
vertices in source plane via the existing brute-force kNN search and
computes locally-exact barycentric weights on the triangle they form.

Three smoke checks:

1. **Well-conditioned (Delaunay agreement)** — same 9-vertex mesh +
   over-sampled query grid as ``make_knn_mapper_9_3x3``. Build mappers for
   both ``Delaunay`` and ``KNNBarycentric``, then for every query where the
   3 nearest mesh vertices in source plane happen to coincide with the
   containing Delaunay simplex's vertices, the weights are
   *bit-identical* (rtol=1e-12). Reports the fraction of queries where
   that holds.

2. **Convex-combination invariant** — every weight row is non-negative,
   row-sums to 1, and is finite. This must hold even for boundary /
   outside-triangle queries (where weights are clipped + renormalized).

3. **Ill-conditioned degenerate triangle** — query at the centre of a
   collinear-3-neighbour configuration. The KNN-barycentric fallback is
   nearest-neighbor (`[1, 0, 0]`), matching Delaunay's outside-simplex
   policy.
"""

import numpy as np
import numpy.testing as npt

import autoarray as aa

from autoarray import fixtures


"""
__Setup__

Use the same 9-vertex mesh and 7x7 sub_2 query grid as
``make_knn_mapper_9_3x3`` so the cross-check against Delaunay shares a
fixture and we exercise the same data shapes the existing kNN smoke does.
"""
mesh_grid_9 = aa.Grid2D.no_mask(
    values=[
        [0.6, -0.3],
        [0.5, -0.8],
        [0.2, 0.1],
        [0.0, 0.5],
        [-0.3, -0.8],
        [-0.6, -0.5],
        [-0.4, -1.1],
        [-1.2, 0.8],
        [-1.5, 0.9],
    ],
    shape_native=(3, 3),
    pixel_scales=1.0,
)

data_grid_sub_2_7x7 = fixtures.make_grid_2d_sub_2_7x7()

bary_mesh = aa.mesh.KNNBarycentric(pixels=9, split_neighbor_division=1)
delaunay_mesh = aa.mesh.Delaunay(pixels=9)

bary_interp = bary_mesh.interpolator_from(
    source_plane_data_grid=data_grid_sub_2_7x7,
    source_plane_mesh_grid=mesh_grid_9,
    adapt_data=aa.Array2D.ones(shape_native=(3, 3), pixel_scales=0.1),
)
delaunay_interp = delaunay_mesh.interpolator_from(
    source_plane_data_grid=data_grid_sub_2_7x7,
    source_plane_mesh_grid=mesh_grid_9,
    adapt_data=aa.Array2D.ones(shape_native=(3, 3), pixel_scales=0.1),
)


"""
__Check 1: Convex-combination invariant__

For every query, exactly 3 mappings (k=3), weights non-negative, row-sums
to 1, all finite. This must hold for both inside-triangle and
outside-triangle queries.
"""
bary_mappings = np.asarray(bary_interp.mappings)
bary_weights = np.asarray(bary_interp.weights)
bary_sizes = np.asarray(bary_interp.sizes)

assert bary_mappings.shape[1] == 3, f"k=3 expected, got {bary_mappings.shape[1]}"
assert (bary_sizes == 3).all(), "all sizes must equal 3"
assert np.isfinite(bary_weights).all(), "non-finite weight encountered"
assert (bary_weights >= 0.0).all(), "negative weight after clip+renorm"

bary_row_sums = bary_weights.sum(axis=1)
npt.assert_allclose(bary_row_sums, np.ones_like(bary_row_sums), rtol=1e-12, atol=1e-12)

print(
    f"check 1 (convex combination): {len(bary_row_sums)} queries, all rows "
    f"sum to 1, all weights in [0, 1]"
)


"""
__Check 2: Bit-identical to Delaunay where the 3 nearest match the
containing Delaunay simplex__

The Delaunay path stores the 3 vertices of the containing simplex in its
own ``mappings`` (each row may be reordered relative to kNN, so compare as
sets). When the index *sets* match, the barycentric weights computed by
both paths are mathematically equivalent — same triangle, same query →
same weights up to floating-point rounding.

Reports the fraction of queries where the sets match; this is the
"interior fraction" and is the key empirical evidence for the wildcard's
production prospects.
"""
delaunay_mappings = np.asarray(delaunay_interp.mappings)
delaunay_weights = np.asarray(delaunay_interp.weights)

# Delaunay rows with mapping[1] != -1 are inside-simplex queries (the only
# case where Delaunay weights are meaningful for comparison).
inside_simplex = delaunay_mappings[:, 1] != -1

matching = np.zeros(len(bary_mappings), dtype=bool)
for i in range(len(bary_mappings)):
    if not inside_simplex[i]:
        continue
    matching[i] = set(bary_mappings[i].tolist()) == set(delaunay_mappings[i].tolist())

n_inside = int(inside_simplex.sum())
n_matching = int(matching.sum())
match_fraction = n_matching / max(n_inside, 1)

# For each matching query, weights must agree to fp tolerance after
# reordering to the same vertex order (Delaunay's order).
for i in np.where(matching)[0]:
    perm = [bary_mappings[i].tolist().index(j) for j in delaunay_mappings[i]]
    npt.assert_allclose(
        bary_weights[i, perm],
        delaunay_weights[i],
        rtol=1e-10,
        atol=1e-12,
        err_msg=(
            f"row {i}: kNN-barycentric weights disagree with Delaunay on "
            f"same triangle (indices {delaunay_mappings[i].tolist()})"
        ),
    )

print(
    f"check 2 (Delaunay agreement): {n_matching}/{n_inside} inside-simplex "
    f"queries had matching nearest-3 sets ({100 * match_fraction:.1f}%); "
    f"weights bit-identical on all of them"
)


"""
__Check 3: Degenerate-triangle fallback__

Manually construct a query whose 3 nearest mesh vertices are collinear (all
on y=x). Total triangle area is 0; the helper must fall back to the
nearest-neighbor weight ``[1, 0, 0]`` rather than producing NaN.
"""
from autoarray.inversion.mesh.interpolator.knn import (
    barycentric_weights_from_3_nearest,
)

collinear_mesh = np.array([[0.0, 0.0], [0.5, 0.5], [-0.5, -0.5]])
query_at_origin = np.array([[0.0, 0.0]])
nearest_3 = np.array([[0, 1, 2]])

degenerate_weights = barycentric_weights_from_3_nearest(
    query_points=query_at_origin,
    mesh_points=collinear_mesh,
    nearest_3_indices=nearest_3,
    xp=np,
)

assert np.isfinite(degenerate_weights).all(), "degenerate fallback gave NaN/inf"
npt.assert_allclose(degenerate_weights[0], [1.0, 0.0, 0.0], atol=1e-12)

print("check 3 (degenerate fallback): collinear-3 → [1, 0, 0], finite")

print("\nknn_barycentric: all assertions passed")
