"""DelaunayNN fixed-shape cap audit on production-like lensing geometries.

The Sibson implementation requires static JAX shapes for the Delaunay
insertion cavity and compact natural-neighbour stencil. This script tests the
candidate cap of 16 against:

* a local 25-step trajectory around a normal Isothermal + shear mass model;
* a broad deterministic sample of mass ellipticity, orientation, Einstein
  radius, centre and shear;
* both data-grid interpolation and the 4*N split points used by split
  regularization.

The image-plane mesh is a 1,200-vertex Hilbert mesh built from an arc-like
adapt image. Every mass model ray-traces the same image-plane data and mesh
coordinates before qhull and the pure-JAX Sibson calculation run. A 64-entry
reference pass records the untruncated distributions, then the worst geometry
is rerun at caps 16 and 32.

Override ``DELAUNAY_NN_CAP_RANDOM_SAMPLES`` for a shorter exploratory run.
The fixed stress geometry is always included, so the cap-16 regression remains
covered even in a reduced run.

ENV: jax full_datasets
"""

import os

import autoarray as aa
import autolens as al
import jax
import jax.numpy as jnp
import numpy as np
from autoarray.inversion.mesh.interpolator.sibson import jax_delaunay_nn

jax.config.update("jax_enable_x64", True)

MESH_POINTS = int(os.environ.get("DELAUNAY_NN_CAP_MESH_POINTS", "1200"))
RANDOM_SAMPLES = int(os.environ.get("DELAUNAY_NN_CAP_RANDOM_SAMPLES", "75"))
REFERENCE_CAP = 64
CANDIDATE_CAP = 16
INTERMEDIATE_CAP = 24
DEFAULT_CAP = 32


def mass_parameter_sets():
    """Return local perturbations, a broad prior sample, and a fixed stress case."""
    parameters = []

    for offset in np.linspace(-1.0, 1.0, 25):
        parameters.append(
            {
                "einstein_radius": 1.6 + 0.12 * offset,
                "axis_ratio": 0.8 + 0.06 * np.sin(np.pi * offset),
                "angle": 45.0 + 18.0 * offset,
                "centre": (0.025 * offset, -0.02 * offset),
                "gamma_1": 0.04 + 0.015 * offset,
                "gamma_2": -0.025 + 0.01 * offset,
                "family": "local",
            }
        )

    rng = np.random.default_rng(99)
    for _ in range(RANDOM_SAMPLES):
        parameters.append(
            {
                "einstein_radius": rng.uniform(0.9, 2.2),
                "axis_ratio": rng.uniform(0.4, 1.0),
                "angle": rng.uniform(0.0, 180.0),
                "centre": tuple(rng.uniform(-0.15, 0.15, 2)),
                "gamma_1": rng.uniform(-0.12, 0.12),
                "gamma_2": rng.uniform(-0.12, 0.12),
                "family": "broad",
            }
        )

    # Deterministically recovered by the broad sweep as its largest cavity.
    # Keep it explicit so reduced developer runs still prove cap 16 is unsafe.
    parameters.append(
        {
            "einstein_radius": 1.410939720834265,
            "axis_ratio": 0.47599146027345235,
            "angle": 143.25245036718673,
            "centre": (0.12529337, -0.10065187),
            "gamma_1": -0.08283306320251471,
            "gamma_2": 0.11855487676798926,
            "family": "fixed_stress",
        }
    )
    return parameters


def traced_grids_from(parameters, image_grid, image_mesh_grid):
    mass = al.mp.Isothermal(
        centre=parameters["centre"],
        einstein_radius=parameters["einstein_radius"],
        ell_comps=al.convert.ell_comps_from(
            axis_ratio=parameters["axis_ratio"],
            angle=parameters["angle"],
        ),
    )
    shear = al.mp.ExternalShear(
        gamma_1=parameters["gamma_1"],
        gamma_2=parameters["gamma_2"],
    )
    tracer = al.Tracer(
        galaxies=[
            al.Galaxy(redshift=0.5, mass=mass, shear=shear),
            al.Galaxy(redshift=1.0),
        ]
    )
    source_grid = tracer.traced_grid_2d_list_from(grid=image_grid)[-1]
    source_mesh_grid = tracer.traced_grid_2d_list_from(grid=image_mesh_grid)[-1]
    return np.asarray(source_mesh_grid), np.asarray(source_grid)


def table_function(cap):
    return jax.jit(
        lambda points, queries: jax_delaunay_nn(
            points,
            queries,
            max_cavity_triangles=cap,
            max_neighbors=cap,
            query_chunk=256,
        )
    )


mask = al.Mask2D.circular(
    shape_native=(81, 81),
    pixel_scales=0.08,
    radius=3.0,
)
image_grid = al.Grid2D.from_mask(mask=mask)
grid_y, grid_x = np.asarray(image_grid.array).T
radius = np.sqrt((grid_y - 0.1) ** 2 + (grid_x + 0.1) ** 2)
angle = np.arctan2(grid_y, grid_x)
adapt_values = np.exp(-0.5 * ((radius - 1.3) / 0.18) ** 2) * (
    1.0 + 0.4 * np.cos(2.0 * angle)
)
adapt_image = al.Array2D(values=adapt_values, mask=mask)

image_mesh = al.image_mesh.Hilbert(
    pixels=MESH_POINTS,
    weight_power=1.0,
    weight_floor=0.01,
)
image_mesh_grid = aa.Grid2DIrregular(
    image_mesh.image_plane_mesh_grid_from(mask=mask, adapt_data=adapt_image)
)

reference = table_function(REFERENCE_CAP)
main_sizes = []
main_cavity_sizes = []
split_sizes = []
split_cavity_sizes = []
worst = None

parameters_list = mass_parameter_sets()
for sample_index, parameters in enumerate(parameters_list):
    points, queries = traced_grids_from(
        parameters=parameters,
        image_grid=image_grid,
        image_mesh_grid=image_mesh_grid,
    )
    outputs = reference(jnp.asarray(points), jnp.asarray(queries))
    jax.block_until_ready(outputs)

    sample_main_sizes = np.asarray(outputs[3])
    sample_main_cavity_sizes = np.asarray(outputs[9])
    sample_split_sizes = np.asarray(outputs[7])
    sample_split_cavity_sizes = np.asarray(outputs[12])

    assert not np.asarray(outputs[10]).any(), "reference main cavity overflow"
    assert not np.asarray(outputs[11]).any(), "reference main degeneracy"
    assert not np.asarray(outputs[13]).any(), "reference split cavity overflow"
    assert not np.asarray(outputs[14]).any(), "reference split degeneracy"

    main_sizes.append(sample_main_sizes)
    main_cavity_sizes.append(sample_main_cavity_sizes)
    split_sizes.append(sample_split_sizes)
    split_cavity_sizes.append(sample_split_cavity_sizes)

    sample_maximum = max(
        sample_main_sizes.max(),
        sample_main_cavity_sizes.max(),
        sample_split_sizes.max(),
        sample_split_cavity_sizes.max(),
    )
    if worst is None or sample_maximum > worst[0]:
        worst = (sample_maximum, sample_index, parameters, points, queries)

main_sizes = np.concatenate(main_sizes)
main_cavity_sizes = np.concatenate(main_cavity_sizes)
split_sizes = np.concatenate(split_sizes)
split_cavity_sizes = np.concatenate(split_cavity_sizes)

observed = {
    "main_neighbors": int(main_sizes.max()),
    "main_cavity": int(main_cavity_sizes.max()),
    "split_neighbors": int(split_sizes.max()),
    "split_cavity": int(split_cavity_sizes.max()),
}

assert max(observed.values()) > CANDIDATE_CAP, (
    "The fixed stress geometry should demonstrate that cap 16 truncates a "
    f"production-like stencil; observed {observed}"
)
assert max(observed.values()) <= DEFAULT_CAP, (
    f"The current cap 32 needs increasing for this audit; observed {observed}"
)

_, worst_index, worst_parameters, worst_points, worst_queries = worst
cap_16_outputs = table_function(CANDIDATE_CAP)(
    jnp.asarray(worst_points), jnp.asarray(worst_queries)
)
jax.block_until_ready(cap_16_outputs)
cap_24_outputs = table_function(INTERMEDIATE_CAP)(
    jnp.asarray(worst_points), jnp.asarray(worst_queries)
)
jax.block_until_ready(cap_24_outputs)
cap_32_outputs = table_function(DEFAULT_CAP)(
    jnp.asarray(worst_points), jnp.asarray(worst_queries)
)
jax.block_until_ready(cap_32_outputs)

cap_16_overflow = int(np.asarray(cap_16_outputs[10]).sum()) + int(
    np.asarray(cap_16_outputs[13]).sum()
)
cap_24_overflow = int(np.asarray(cap_24_outputs[10]).sum()) + int(
    np.asarray(cap_24_outputs[13]).sum()
)
assert cap_16_overflow > 0, "cap 16 did not report the expected truncation"
assert cap_24_overflow > 0, "cap 24 did not report the expected truncation"
assert (
    np.isnan(np.asarray(cap_16_outputs[4])).any()
    or np.isnan(np.asarray(cap_16_outputs[8])).any()
)
assert not np.asarray(cap_32_outputs[10]).any()
assert not np.asarray(cap_32_outputs[13]).any()

print(f"mass models audited: {len(parameters_list)}")
print(f"mesh/data rows: {MESH_POINTS}/{image_grid.shape[0]}")
print(f"observed maxima: {observed}")
print(
    "main-neighbor percentiles (99, 99.9, 99.99): "
    f"{np.percentile(main_sizes, [99.0, 99.9, 99.99])}"
)
print(
    "split-neighbor percentiles (99, 99.9, 99.99): "
    f"{np.percentile(split_sizes, [99.0, 99.9, 99.99])}"
)
print(
    "rows exceeding cap 16: "
    f"main_neighbors={int((main_sizes > 16).sum())}, "
    f"main_cavities={int((main_cavity_sizes > 16).sum())}, "
    f"split_neighbors={int((split_sizes > 16).sum())}, "
    f"split_cavities={int((split_cavity_sizes > 16).sum())}"
)
print(
    f"worst sample: index={worst_index}, family={worst_parameters['family']}, "
    f"cap_16_overflow_rows={cap_16_overflow}, "
    f"cap_24_overflow_rows={cap_24_overflow}"
)
print("PASS: caps 16 and 24 are too low; cap 32 covers this production-like audit.")
