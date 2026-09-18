"""
Correctness, jit/vmap, autodiff and timing gate for the JAX Delaunay point locator.

``pix_indexes_delaunay_walk_from`` replaces ``scipy.spatial.Delaunay.find_simplex``
on the JAX likelihood path with a visibility walk: a nearest-vertex seed
(chunked only to bound the ``(chunk, N)`` argmin under vmap) followed by a
``lax.while_loop`` that exits as soon as every query is located or has left the
convex hull. The walk is the exact algorithm ``find_simplex`` itself uses, so
"close enough" is not the contract — the located simplex must be the one qhull
reports, and an outside-hull query must fall back to the same nearest vertex
``scipy_delaunay`` assigns through its KDTree.

The checks below are the ones the early-exit rewrite can break:

* parity against ``find_simplex`` + ``cKDTree`` on uniform, blob-ring and
  production-like lensing geometries, for the data grid and the 4*N split
  points, allowing only shared-edge ties (the returned simplex must still
  contain the query);
* ``jax.jit`` equals eager on a query count that is *not* a multiple of the
  seed chunk, and on an odd short query set — the padding the chunked argmin
  applies must never leak into the result;
* a jitted ``vmap`` over a batch of traced query sets equals the per-member
  results, since the ``while_loop`` runs to the slowest lane in a batch;
* ``jax.grad`` of a barycentric-weight scalar with respect to the query
  coordinates runs at all (``lax.while_loop`` has no reverse-mode rule, so the
  locator ``stop_gradient``-wraps its float inputs) and matches central finite
  differences;
* the warm per-call time of the jitted locator on the lensing geometry is
  printed for the record; nothing gates on it.

Override ``DELAUNAY_WALK_MESH_POINTS`` and ``DELAUNAY_WALK_MASS_MODELS`` for a
shorter local probe. The fixed stress geometry is always the first mass model,
so a reduced run keeps the production-like parity check.

__Env__

Test-harness configuration (PyAutoHands docs/env_profile_redesign.md §10).
Every check gates a JAX code path — the chunked seed argmin, the ``while_loop``
walk, its jit/vmap round-trips and the ``stop_gradient`` gradient contract — so
JAX must stay enabled. Parity is measured on a production-like 1,500-vertex
Hilbert mesh ray-traced through full mass models, and the seed chunk is 1,024
wide, so the SMALL_DATASETS cap must stay off or the chunk boundary and the
walk's step distribution are never exercised.

ENV: jax full_datasets
"""

import os
import time

import autoarray as aa
import autolens as al
import jax
import jax.numpy as jnp
import numpy as np
from scipy.spatial import Delaunay, cKDTree

from autoarray.inversion.mesh.interpolator.delaunay import (
    DELAUNAY_LOCATE_CHUNK,
    _jax_delaunay_tables,
    jax_delaunay,
    pix_indexes_delaunay_walk_from,
    pixel_weights_delaunay_from,
)

jax.config.update("jax_enable_x64", True)

MESH_POINTS = int(os.environ.get("DELAUNAY_WALK_MESH_POINTS", "1500"))
MASS_MODELS = int(os.environ.get("DELAUNAY_WALK_MASS_MODELS", "3"))
TIMING_REPEATS = int(os.environ.get("DELAUNAY_WALK_REPEATS", "3"))

FAILURES = []


def check(name, passed, extra=""):
    """Print a PASS/FAIL line and record the failure for the closing assert."""
    print(f"  [{'PASS' if passed else 'FAIL'}] {name} {extra}".rstrip())
    if not passed:
        FAILURES.append(name)


def adaptive_mesh(count, rng):
    """A central blob plus a noisy ring — the source-plane-like mesh used by
    ``delaunay_nn.py``, with a hull that queries readily fall outside of."""
    blob_count = count // 2
    blob = rng.normal(size=(blob_count, 2)) * 0.15
    angle = rng.uniform(0.0, 2.0 * np.pi, size=count - blob_count)
    radius = 1.0 + rng.normal(size=count - blob_count) * 0.12
    ring = np.stack([radius * np.cos(angle), radius * np.sin(angle)], axis=1)
    return np.concatenate([blob, ring])


def walk_tables(points):
    return tuple(_jax_delaunay_tables(jnp.asarray(points)))


def locate(query_points, points, tables=None, jit=False, return_simplex_indexes=False):
    simplices_padded, simplex_neighbors, vertex_simplex = tables or walk_tables(points)

    def located(query, mesh):
        return pix_indexes_delaunay_walk_from(
            query_points=query,
            points=mesh,
            simplices_padded=simplices_padded,
            simplex_neighbors=simplex_neighbors,
            vertex_simplex=vertex_simplex,
            xp=jnp,
            return_simplex_indexes=return_simplex_indexes,
        )

    if jit:
        located = jax.jit(located)
    return located(jnp.asarray(query_points), jnp.asarray(points))


def barycentric_coordinates(points, query, mappings):
    """Barycentric coordinates of each query in the triangle it was mapped to."""
    a, b, c = points[mappings[:, 0]], points[mappings[:, 1]], points[mappings[:, 2]]

    def cross(u, v):
        return u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]

    denominator = cross(b - a, c - a)
    return (
        np.stack(
            [
                cross(b - query, c - query),
                cross(c - query, a - query),
                cross(a - query, b - query),
            ],
            axis=1,
        )
        / denominator[:, None]
    )


def parity(label, points, query, mappings):
    """Compare a located mapping table against ``find_simplex`` + ``cKDTree``.

    Rows are compared as sorted vertex sets, not triangle ids: a query on a
    shared edge belongs to both incident triangles and either answer is exact.
    Such a tie is only accepted when the returned triangle still contains the
    query (minimum barycentric coordinate >= -1e-9). Outside-hull rows must
    reproduce the KDTree nearest vertex exactly, in the ``[v, -1, -1]`` form
    ``scipy_delaunay`` uses.
    """
    points = np.asarray(points)
    query = np.asarray(query)
    mappings = np.asarray(mappings)

    triangulation = Delaunay(points)
    simplex_indexes = triangulation.find_simplex(query)
    inside = simplex_indexes >= 0
    _, nearest = cKDTree(points).query(query, k=1)

    expected = np.full_like(mappings, -1)
    expected[inside] = triangulation.simplices[simplex_indexes[inside]]
    expected[~inside, 0] = nearest[~inside]

    identical = (np.sort(mappings, axis=1) == np.sort(expected, axis=1)).all(axis=1)

    outside_exact = bool(
        (mappings[~inside, 0] == nearest[~inside]).all()
        and (mappings[~inside, 1:] == -1).all()
    )

    tie_rows = np.where(inside & ~identical)[0]
    ties_contain_query = True
    if tie_rows.size:
        tie_mappings = mappings[tie_rows]
        ties_contain_query = bool((tie_mappings[:, 1:] >= 0).all())
        if ties_contain_query:
            coordinates = barycentric_coordinates(points, query[tie_rows], tie_mappings)
            ties_contain_query = bool((coordinates.min(axis=1) >= -1.0e-9).all())

    check(
        f"parity {label}",
        outside_exact and ties_contain_query,
        f"(exact rows {identical.mean():.6f}, {tie_rows.size} shared-edge ties, "
        f"{int((~inside).sum())} outside hull)",
    )
    return mappings


print(f"device: {jax.devices()[0]}")
print("=== 1a. parity vs find_simplex + cKDTree on synthetic meshes ===")

rng = np.random.default_rng(2)
uniform_points = rng.uniform(-1.0, 1.0, size=(400, 2))
uniform_query = rng.uniform(-1.2, 1.2, size=(3000, 2))
parity(
    "uniform N=400 Q=3000",
    uniform_points,
    uniform_query,
    locate(uniform_query, uniform_points),
)

rng = np.random.default_rng(4)
blob_points = adaptive_mesh(400, rng)
blob_query = np.concatenate([adaptive_mesh(2800, rng), rng.normal(size=(200, 2)) * 1.6])
parity(
    "blob-ring N=400 Q=3000", blob_points, blob_query, locate(blob_query, blob_points)
)

rng = np.random.default_rng(21)
large_points = adaptive_mesh(1500, rng)
large_query = np.concatenate(
    [adaptive_mesh(15361, rng), rng.normal(size=(613, 2)) * 1.8]
)
large_tables = walk_tables(large_points)
parity(
    f"blob-ring N=1500 Q={large_query.shape[0]}",
    large_points,
    large_query,
    locate(large_query, large_points, tables=large_tables),
)

print("=== 1b. parity on lensing geometries (data grid + 4N split points) ===")

# The image-plane setup of ``delaunay_nn_caps.py``: an arc-like adapt image
# drives a Hilbert mesh, and every mass model ray-traces the same image-plane
# data and mesh coordinates into the source plane the walk actually runs in.
mask = al.Mask2D.circular(shape_native=(81, 81), pixel_scales=0.08, radius=3.0)
image_grid = al.Grid2D.from_mask(mask=mask)
grid_y, grid_x = np.asarray(image_grid.array).T
adapt_radius = np.sqrt((grid_y - 0.1) ** 2 + (grid_x + 0.1) ** 2)
adapt_angle = np.arctan2(grid_y, grid_x)
adapt_image = al.Array2D(
    values=np.exp(-0.5 * ((adapt_radius - 1.3) / 0.18) ** 2)
    * (1.0 + 0.4 * np.cos(2.0 * adapt_angle)),
    mask=mask,
)
image_mesh = al.image_mesh.Hilbert(
    pixels=MESH_POINTS, weight_power=1.0, weight_floor=0.01
)
image_mesh_grid = aa.Grid2DIrregular(
    image_mesh.image_plane_mesh_grid_from(mask=mask, adapt_data=adapt_image)
)

# The fixed stress geometry ``delaunay_nn_caps.py`` recovered as the worst case
# of its broad prior sweep, plus two deterministic samples from that sweep's
# ranges. Kept explicit so a reduced run is still the production-like case.
MASS_PARAMETERS = [
    {
        "einstein_radius": 1.410939720834265,
        "axis_ratio": 0.47599146027345235,
        "angle": 143.25245036718673,
        "centre": (0.12529337, -0.10065187),
        "gamma_1": -0.08283306320251471,
        "gamma_2": 0.11855487676798926,
    },
    {
        "einstein_radius": 1.6,
        "axis_ratio": 0.8,
        "angle": 45.0,
        "centre": (0.0, 0.0),
        "gamma_1": 0.04,
        "gamma_2": -0.025,
    },
    {
        "einstein_radius": 2.05,
        "axis_ratio": 0.95,
        "angle": 12.0,
        "centre": (-0.11, 0.09),
        "gamma_1": 0.10,
        "gamma_2": 0.07,
    },
    {
        "einstein_radius": 0.95,
        "axis_ratio": 0.42,
        "angle": 172.0,
        "centre": (0.14, 0.13),
        "gamma_1": -0.11,
        "gamma_2": -0.09,
    },
][:MASS_MODELS]


def traced_grids_from(parameters):
    tracer = al.Tracer(
        galaxies=[
            al.Galaxy(
                redshift=0.5,
                mass=al.mp.Isothermal(
                    centre=parameters["centre"],
                    einstein_radius=parameters["einstein_radius"],
                    ell_comps=al.convert.ell_comps_from(
                        axis_ratio=parameters["axis_ratio"],
                        angle=parameters["angle"],
                    ),
                ),
            ),
            al.Galaxy(redshift=1.0),
        ],
        # The external shear is a property of the system, so it is an `al.MassField` in `fields=`.
        fields=[
            al.MassField(
                redshift=0.5,
                shear=al.mp.ExternalShear(
                    gamma_1=parameters["gamma_1"], gamma_2=parameters["gamma_2"]
                ),
            )
        ],
    )
    source_mesh = tracer.traced_grid_2d_list_from(grid=image_mesh_grid)[-1]
    source_data = tracer.traced_grid_2d_list_from(grid=image_grid)[-1]
    return np.asarray(source_mesh), np.asarray(source_data)


jax_delaunay_jit = jax.jit(jax_delaunay)
lensing_geometries = []

for index, parameters in enumerate(MASS_PARAMETERS):
    source_mesh, source_data = traced_grids_from(parameters)
    outputs = jax_delaunay_jit(jnp.asarray(source_mesh), jnp.asarray(source_data))
    jax.block_until_ready(outputs)
    _, _, mappings, split_points, split_mappings, _ = outputs
    lensing_geometries.append((source_mesh, source_data))

    parity(
        f"lensing model {index} data grid N={source_mesh.shape[0]} "
        f"Q={source_data.shape[0]}",
        source_mesh,
        source_data,
        mappings,
    )
    parity(
        f"lensing model {index} split points Q={4 * source_mesh.shape[0]}",
        source_mesh,
        np.asarray(split_points),
        split_mappings,
    )

print("=== 2. jit == eager, chunk-boundary and odd query counts ===")

assert large_query.shape[0] % DELAUNAY_LOCATE_CHUNK != 0
eager = np.asarray(locate(large_query, large_points, tables=large_tables))
jitted = np.asarray(locate(large_query, large_points, tables=large_tables, jit=True))
check(
    f"jit == eager (Q={large_query.shape[0]}, "
    f"{large_query.shape[0] % DELAUNAY_LOCATE_CHUNK} past the {DELAUNAY_LOCATE_CHUNK} "
    "chunk boundary)",
    bool((eager == jitted).all()),
)

odd_query = adaptive_mesh(1023, np.random.default_rng(31))
odd_eager = np.asarray(locate(odd_query, blob_points))
odd_mappings, odd_simplexes = locate(
    odd_query, blob_points, jit=True, return_simplex_indexes=True
)
odd_mappings = np.asarray(odd_mappings)
check("odd Q=1023 jit == eager", bool((odd_eager == odd_mappings).all()))
check(
    "return_simplex_indexes shapes and dtypes",
    odd_mappings.shape == (1023, 3)
    and odd_mappings.dtype == np.int32
    and np.asarray(odd_simplexes).shape == (1023,)
    and np.asarray(odd_simplexes).dtype == np.int32,
    f"(mappings {odd_mappings.shape} {odd_mappings.dtype}, "
    f"simplexes {np.asarray(odd_simplexes).shape} {np.asarray(odd_simplexes).dtype})",
)

print("=== 3. jitted vmap over a batch of traced query sets ===")

batch_mesh = jnp.asarray(lensing_geometries[0][0])
batch_queries = jnp.asarray(
    np.stack([source_data for _, source_data in lensing_geometries])
)
batch_tables = walk_tables(np.asarray(batch_mesh))


def locate_one(query):
    return pix_indexes_delaunay_walk_from(
        query_points=query,
        points=batch_mesh,
        simplices_padded=batch_tables[0],
        simplex_neighbors=batch_tables[1],
        vertex_simplex=batch_tables[2],
        xp=jnp,
    )


batched = np.asarray(jax.jit(jax.vmap(locate_one))(batch_queries))
per_member = np.stack([np.asarray(locate_one(query)) for query in batch_queries])
check(
    "jit(vmap) == per-member results",
    bool((batched == per_member).all()),
    f"(batch {batched.shape})",
)

print("=== 4. jax.grad through the barycentric weights ===")

gradient_points = jnp.asarray(blob_points)
gradient_query = jnp.asarray(blob_query[:512])
gradient_tables = walk_tables(blob_points)


def weight_energy(query):
    mappings = pix_indexes_delaunay_walk_from(
        query_points=query,
        points=gradient_points,
        simplices_padded=gradient_tables[0],
        simplex_neighbors=gradient_tables[1],
        vertex_simplex=gradient_tables[2],
        xp=jnp,
    )
    weights = pixel_weights_delaunay_from(
        data_grid=query,
        mesh_grid=gradient_points,
        pix_indexes_for_sub_slim_index=mappings,
        xp=jnp,
    )
    return jnp.sum(weights**2)


try:
    gradient = np.asarray(jax.grad(weight_energy)(gradient_query))
    gradient_ran = bool(np.isfinite(gradient).all())
except Exception as exception:  # noqa: BLE001
    gradient, gradient_ran = None, False
    print(f"      jax.grad raised: {exception!r}")

check(
    "jax.grad w.r.t. query_points runs (no while_loop reverse-mode error)",
    gradient_ran,
)

if gradient_ran:
    step = 1.0e-6
    errors = []
    for index in (17, 128, 400):
        for dimension in (0, 1):
            forward = float(
                weight_energy(gradient_query.at[index, dimension].add(step))
            )
            backward = float(
                weight_energy(gradient_query.at[index, dimension].add(-step))
            )
            finite_difference = (forward - backward) / (2.0 * step)
            autodiff = float(gradient[index, dimension])
            errors.append(
                (
                    index,
                    dimension,
                    autodiff,
                    finite_difference,
                    abs(autodiff - finite_difference)
                    / max(abs(finite_difference), 1.0e-12),
                )
            )
    for index, dimension, autodiff, finite_difference, relative in errors:
        print(
            f"      point {index} dim {dimension}: ad={autodiff: .10e} "
            f"fd={finite_difference: .10e} rel={relative:.2e}"
        )
    check(
        "central finite differences match autodiff at 3 points",
        all(error[4] < 1.0e-6 for error in errors),
        f"(worst rel {max(error[4] for error in errors):.2e})",
    )

print("=== 5. warm locator time on the lensing geometry (reported, not gated) ===")

timing_mesh, timing_data = lensing_geometries[0]
timing_tables = walk_tables(timing_mesh)
timing_split = np.asarray(
    jax_delaunay_jit(jnp.asarray(timing_mesh), jnp.asarray(timing_data))[3]
)
timing_query = jnp.asarray(np.concatenate([timing_data, timing_split]))
timing_mesh = jnp.asarray(timing_mesh)


def timed_locate(query, mesh):
    return pix_indexes_delaunay_walk_from(
        query_points=query,
        points=mesh,
        simplices_padded=timing_tables[0],
        simplex_neighbors=timing_tables[1],
        vertex_simplex=timing_tables[2],
        xp=jnp,
    )


timed_locate_jit = jax.jit(timed_locate)
start = time.perf_counter()
jax.block_until_ready(timed_locate_jit(timing_query, timing_mesh))
compile_and_first_s = time.perf_counter() - start

samples = []
for _ in range(TIMING_REPEATS):
    start = time.perf_counter()
    jax.block_until_ready(timed_locate_jit(timing_query, timing_mesh))
    samples.append(time.perf_counter() - start)

print(
    f"      N={timing_mesh.shape[0]} mesh points, Q={timing_query.shape[0]} queries "
    f"(data grid + 4N split points)"
)
print(
    f"      compile+first={compile_and_first_s:.6f}s "
    f"warm_median={float(np.median(samples)):.6f}s runs={samples}"
)

print()
assert not FAILURES, f"delaunay walk checks failed: {FAILURES}"
print("PASS: the JAX Delaunay walk reproduces find_simplex under jit, vmap and grad.")
