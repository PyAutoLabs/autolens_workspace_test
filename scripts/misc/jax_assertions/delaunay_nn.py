"""Correctness, reconstruction, autodiff, and performance gate for DelaunayNN.

The first check exercises the public ``aa.mesh.DelaunayNN`` through a Mapper
and Inversion and compares its reconstructed source with the otherwise
identical barycentric ``aa.mesh.Delaunay`` inversion. The remaining checks use
synthetic production-sized arrays so they can be timed on an accelerator:

* public mesh/interpolator, split-regularization, and source reconstruction;
* JIT execution and finite, normalized Sibson weights;
* exact linear precision and its analytic query-coordinate gradient;
* continuity of values and gradients through a Delaunay diagonal flip;
* a jitted ``vmap`` through independent qhull callbacks;
* warm runtime against the current barycentric Delaunay interpolation.

Override ``SIBSON_POINTS``, ``SIBSON_QUERIES`` and ``SIBSON_REPEATS`` for a
short local probe or a larger accelerator run.

ENV: jax full_datasets
"""

import os
import time

import autoarray as aa
import jax
import jax.numpy as jnp
import numpy as np
from autoarray import fixtures
from autoarray.inversion.mesh.interpolator.delaunay import (
    _jax_delaunay_tables,
    jax_delaunay,
    pix_indexes_delaunay_walk_from,
    pixel_weights_delaunay_from,
)
from autoarray.inversion.mesh.interpolator.sibson import (
    InterpolatorDelaunayNN,
    jax_delaunay_nn,
    jax_sibson,
)

jax.config.update("jax_enable_x64", True)

POINT_COUNT = int(os.environ.get("SIBSON_POINTS", "1200"))
QUERY_COUNT = int(os.environ.get("SIBSON_QUERIES", "15000"))
REPEATS = int(os.environ.get("SIBSON_REPEATS", "5"))
MAX_CAVITY_TRIANGLES = int(os.environ.get("SIBSON_CAVITY", "32"))
MAX_NEIGHBORS = int(os.environ.get("SIBSON_NEIGHBORS", "32"))
QUERY_CHUNK = int(os.environ.get("SIBSON_CHUNK", "256"))


def mapper_from(mesh, mesh_grid, data_grid):
    interpolator = mesh.interpolator_from(
        source_plane_data_grid=data_grid,
        source_plane_mesh_grid=mesh_grid,
        adapt_data=aa.Array2D.ones(shape_native=(3, 3), pixel_scales=0.1),
    )
    return aa.Mapper(
        interpolator=interpolator,
        regularization=aa.reg.Constant(coefficient=1.0),
        image_plane_mesh_grid=aa.Grid2D.uniform(shape_native=(3, 3), pixel_scales=0.1),
    )


# Public integration path: the same dataset, source-plane vertices and
# regularization are inverted with Delaunay and DelaunayNN. The source vectors
# need not be identical because the interpolation bases differ, but a smooth
# reconstruction should be numerically very close.
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
data_grid = fixtures.make_grid_2d_sub_2_7x7()
dataset = fixtures.make_masked_imaging_7x7()

delaunay_mapper = mapper_from(aa.mesh.Delaunay(pixels=9), mesh_grid_9, data_grid)
delaunay_nn_mesh = aa.mesh.DelaunayNN(pixels=9)
delaunay_nn_mapper = mapper_from(delaunay_nn_mesh, mesh_grid_9, data_grid)

assert delaunay_nn_mesh.interpolator_cls is InterpolatorDelaunayNN
assert isinstance(delaunay_nn_mapper.interpolator, InterpolatorDelaunayNN)
assert not np.asarray(delaunay_nn_mapper.interpolator.delaunay.overflow).any()
assert not np.asarray(delaunay_nn_mapper.interpolator.delaunay.degenerate).any()
assert not np.asarray(delaunay_nn_mapper.interpolator.delaunay.split_overflow).any()
assert not np.asarray(delaunay_nn_mapper.interpolator.delaunay.split_degenerate).any()

delaunay_inversion = aa.Inversion(
    dataset=dataset,
    linear_obj_list=[delaunay_mapper],
)
delaunay_nn_inversion = aa.Inversion(
    dataset=dataset,
    linear_obj_list=[delaunay_nn_mapper],
)

delaunay_source = np.asarray(delaunay_inversion.reconstruction)
delaunay_nn_source = np.asarray(delaunay_nn_inversion.reconstruction)
delaunay_source_relative_l2 = np.linalg.norm(
    delaunay_nn_source - delaunay_source
) / np.linalg.norm(delaunay_source)
delaunay_source_correlation = np.corrcoef(delaunay_nn_source, delaunay_source)[0, 1]

assert delaunay_source_relative_l2 < 5.0e-4
assert delaunay_source_correlation > 0.995


def public_interpolator_objective(mesh_points):
    interpolator = delaunay_nn_mesh.interpolator_from(
        source_plane_data_grid=data_grid,
        source_plane_mesh_grid=aa.Grid2DIrregular(values=mesh_points, xp=jnp),
        xp=jnp,
    )
    source_values = jnp.linspace(0.2, 1.8, mesh_points.shape[0])
    safe_mappings = jnp.maximum(interpolator.mappings, 0)
    mapped = source_values[safe_mappings] * interpolator.weights
    return jnp.sum(mapped**2)


public_value, public_mesh_gradient = jax.jit(
    jax.value_and_grad(public_interpolator_objective)
)(jnp.asarray(mesh_grid_9.array))
assert np.isfinite(float(public_value))
assert np.isfinite(np.asarray(public_mesh_gradient)).all()
assert float(jnp.linalg.norm(public_mesh_gradient)) > 0.0

# Exercise the public implementation's full JAX table contract, including
# the 4*N split points used by split regularization.
full_tables_jit = jax.jit(
    lambda mesh_points, queries: jax_delaunay_nn(
        mesh_points,
        queries,
        max_cavity_triangles=MAX_CAVITY_TRIANGLES,
        max_neighbors=MAX_NEIGHBORS,
        query_chunk=QUERY_CHUNK,
    )
)
full_tables = full_tables_jit(
    jnp.asarray(mesh_grid_9.array),
    jnp.asarray(data_grid.over_sampled.array),
)
jax.block_until_ready(full_tables)
assert full_tables[6].shape == (4 * mesh_grid_9.shape[0], MAX_NEIGHBORS)
assert not np.asarray(full_tables[10]).any()
assert not np.asarray(full_tables[11]).any()
assert not np.asarray(full_tables[13]).any()
assert not np.asarray(full_tables[14]).any()


def full_table_objective(mesh_points, queries):
    tables = jax_delaunay_nn(
        mesh_points,
        queries,
        max_cavity_triangles=MAX_CAVITY_TRIANGLES,
        max_neighbors=MAX_NEIGHBORS,
        query_chunk=QUERY_CHUNK,
    )
    source_values = jnp.linspace(0.2, 1.8, mesh_points.shape[0])
    mappings, weights = tables[2], tables[4]
    split_mappings, split_weights = tables[6], tables[8]
    mapped = source_values[jnp.maximum(mappings, 0)] * weights
    split_mapped = source_values[jnp.maximum(split_mappings, 0)] * split_weights
    return jnp.sum(mapped**2) + 0.1 * jnp.sum(split_mapped**2)


_, (mesh_gradient, query_gradient) = jax.jit(
    jax.value_and_grad(full_table_objective, argnums=(0, 1))
)(
    jnp.asarray(mesh_grid_9.array),
    jnp.asarray(data_grid.over_sampled.array),
)
assert np.isfinite(np.asarray(mesh_gradient)).all()
assert np.isfinite(np.asarray(query_gradient)).all()
assert float(jnp.linalg.norm(mesh_gradient)) > 0.0
assert float(jnp.linalg.norm(query_gradient)) > 0.0


def adaptive_mesh(count, rng):
    blob_count = count // 2
    blob = rng.normal(size=(blob_count, 2)) * 0.15
    angle = rng.uniform(0.0, 2.0 * np.pi, size=count - blob_count)
    radius = 1.0 + rng.normal(size=count - blob_count) * 0.12
    ring = np.stack([radius * np.cos(angle), radius * np.sin(angle)], axis=1)
    return np.concatenate([blob, ring])


def barycentric_tables(points, query_points):
    simplices, neighbors, vertex_simplex = _jax_delaunay_tables(points)
    mappings = pix_indexes_delaunay_walk_from(
        query_points=query_points,
        points=points,
        simplices_padded=simplices,
        simplex_neighbors=neighbors,
        vertex_simplex=vertex_simplex,
        xp=jnp,
    )
    weights = pixel_weights_delaunay_from(
        data_grid=query_points,
        mesh_grid=points,
        pix_indexes_for_sub_slim_index=mappings,
        xp=jnp,
    )
    return mappings, weights


def sibson_tables(points, query_points):
    return jax_sibson(
        points,
        query_points,
        max_cavity_triangles=MAX_CAVITY_TRIANGLES,
        max_neighbors=MAX_NEIGHBORS,
        query_chunk=QUERY_CHUNK,
    )[2:]


def delaunay_full_tables(points, query_points):
    return jax_delaunay(points, query_points)


def delaunay_nn_full_tables(points, query_points):
    return jax_delaunay_nn(
        points,
        query_points,
        max_cavity_triangles=MAX_CAVITY_TRIANGLES,
        max_neighbors=MAX_NEIGHBORS,
        query_chunk=QUERY_CHUNK,
    )


def warm_times(function, points, query_points):
    compiled = jax.jit(function)
    start = time.perf_counter()
    output = compiled(points, query_points)
    jax.block_until_ready(output)
    compile_and_first_s = time.perf_counter() - start

    samples = []
    for _ in range(REPEATS):
        start = time.perf_counter()
        output = compiled(points, query_points)
        jax.block_until_ready(output)
        samples.append(time.perf_counter() - start)
    return output, compile_and_first_s, float(np.median(samples)), samples


def flip_points(offset):
    return jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0 + offset]])


flip_query = jnp.array([[0.35, 0.55]])
flip_values = jnp.array([0.0, 1.0, 2.0, 4.0])


def barycentric_flip_value(offset):
    mappings, weights = barycentric_tables(flip_points(offset), flip_query)
    return jnp.sum(flip_values[jnp.maximum(mappings, 0)] * weights)


def sibson_flip_value(offset):
    mappings, _, weights, *_ = sibson_tables(flip_points(offset), flip_query)
    return jnp.sum(flip_values[jnp.maximum(mappings, 0)] * weights)


rng = np.random.default_rng(4)
points = jnp.asarray(adaptive_mesh(POINT_COUNT, rng))
query_points = jnp.asarray(adaptive_mesh(QUERY_COUNT, rng))

barycentric_output, barycentric_compile_s, barycentric_warm_s, barycentric_runs = (
    warm_times(barycentric_tables, points, query_points)
)
sibson_output, sibson_compile_s, sibson_warm_s, sibson_runs = warm_times(
    sibson_tables, points, query_points
)
(
    delaunay_full_output,
    delaunay_full_compile_s,
    delaunay_full_warm_s,
    delaunay_full_runs,
) = warm_times(delaunay_full_tables, points, query_points)
(
    delaunay_nn_full_output,
    delaunay_nn_full_compile_s,
    delaunay_nn_full_warm_s,
    delaunay_nn_full_runs,
) = warm_times(delaunay_nn_full_tables, points, query_points)

mappings, sizes, weights, cavity_sizes, overflow, degenerate = sibson_output
assert not np.asarray(overflow).any(), "Sibson cavity cap overflowed"
assert not np.asarray(degenerate).any(), "Sibson Watson calculation was degenerate"
np.testing.assert_allclose(np.asarray(weights.sum(axis=1)), 1.0, atol=1.0e-11)
assert not np.asarray(delaunay_nn_full_output[10]).any()
assert not np.asarray(delaunay_nn_full_output[11]).any()
assert not np.asarray(delaunay_nn_full_output[13]).any()
assert not np.asarray(delaunay_nn_full_output[14]).any()

# Natural-neighbour coordinates reproduce every affine field exactly.  This
# simultaneously checks the weights and their JAX derivative with respect to
# the moving query coordinates.
source_values = 2.0 * points[:, 0] - 3.0 * points[:, 1] + 0.7
gradient_vertex_indexes = rng.integers(0, POINT_COUNT, size=(64, 3))
gradient_coefficients = rng.dirichlet(np.ones(3), size=64)
gradient_query_points = jnp.asarray(
    np.sum(
        np.asarray(points)[gradient_vertex_indexes] * gradient_coefficients[:, :, None],
        axis=1,
    )
)


def linear_sum(query):
    linear_mappings, _, linear_weights, *_ = sibson_tables(points, query)
    safe_mappings = jnp.maximum(linear_mappings, 0)
    return jnp.sum(source_values[safe_mappings] * linear_weights)


gradient = jax.grad(linear_sum)(gradient_query_points)
expected_gradient = jnp.broadcast_to(jnp.array([2.0, -3.0]), gradient.shape)
np.testing.assert_allclose(
    np.asarray(gradient), np.asarray(expected_gradient), atol=1.0e-9
)

epsilon = 1.0e-7
barycentric_value_and_grad = jax.jit(jax.value_and_grad(barycentric_flip_value))
sibson_value_and_grad = jax.jit(jax.value_and_grad(sibson_flip_value))
barycentric_left = barycentric_value_and_grad(jnp.asarray(-epsilon))
barycentric_right = barycentric_value_and_grad(jnp.asarray(epsilon))
sibson_left = sibson_value_and_grad(jnp.asarray(-epsilon))
sibson_right = sibson_value_and_grad(jnp.asarray(epsilon))

vmap_offsets = jnp.array([-2.0e-3, -1.0e-3, 1.0e-3, 2.0e-3])
vmap_values = jax.jit(jax.vmap(sibson_flip_value))(vmap_offsets)
assert np.isfinite(np.asarray(vmap_values)).all()

assert abs(float(barycentric_left[0] - barycentric_right[0])) > 0.1
np.testing.assert_allclose(
    np.asarray(sibson_left), np.asarray(sibson_right), rtol=1.0e-5, atol=1.0e-6
)

print(f"device: {jax.devices()[0]}")
print(
    "source reconstruction parity: "
    f"relative_l2={delaunay_source_relative_l2:.6e} "
    f"correlation={delaunay_source_correlation:.9f}"
)
print(f"shape: {POINT_COUNT} mesh points x {QUERY_COUNT} queries")
print(
    "barycentric: "
    f"compile+first={barycentric_compile_s:.6f}s "
    f"warm_median={barycentric_warm_s:.6f}s runs={barycentric_runs}"
)
print(
    "sibson: "
    f"compile+first={sibson_compile_s:.6f}s "
    f"warm_median={sibson_warm_s:.6f}s runs={sibson_runs}"
)
print(f"sibson/barycentric warm ratio: {sibson_warm_s / barycentric_warm_s:.3f}x")
print(
    "delaunay full mapper: "
    f"compile+first={delaunay_full_compile_s:.6f}s "
    f"warm_median={delaunay_full_warm_s:.6f}s runs={delaunay_full_runs}"
)
print(
    "delaunay_nn full mapper: "
    f"compile+first={delaunay_nn_full_compile_s:.6f}s "
    f"warm_median={delaunay_nn_full_warm_s:.6f}s runs={delaunay_nn_full_runs}"
)
print(
    "delaunay_nn/delaunay full mapper ratio: "
    f"{delaunay_nn_full_warm_s / delaunay_full_warm_s:.3f}x"
)
print(
    "sibson diagnostics: "
    f"max_cavity={int(np.asarray(cavity_sizes).max())} "
    f"max_neighbors={int(np.asarray(sizes).max())}"
)
print(
    "flip values/gradients: "
    f"barycentric left={tuple(map(float, barycentric_left))} "
    f"right={tuple(map(float, barycentric_right))}; "
    f"sibson left={tuple(map(float, sibson_left))} "
    f"right={tuple(map(float, sibson_right))}"
)
