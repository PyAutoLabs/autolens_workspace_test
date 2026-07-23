"""
Simulator: Point Source
=======================

Simulates a lensed point-source dataset for use by the JAX likelihood
function tests in this folder.

The dataset is saved to `dataset/point_source/simple/` as
`point_dataset_positions_only.json`.

__Model__

 - Lens: `Isothermal` mass (centre near origin, einstein_radius=1.6).
 - Source: Point source at (0.07, 0.07).

__Env__

Test-harness configuration (PyAutoHands docs/env_profile_redesign.md §10).
JAX likelihood functions test JIT compilation; need JAX enabled and full-
size datasets.

ENV: jax full_datasets
"""

from os import path
import numpy as np
import autolens as al

dataset_path = path.join("dataset", "point_source", "simple")

grid = al.Grid2D.uniform(
    shape_native=(100, 100),
    pixel_scales=0.2,
)

lens_galaxy = al.Galaxy(
    redshift=0.5,
    mass=al.mp.Isothermal(
        centre=(0.01, 0.01),
        einstein_radius=1.6,
        ell_comps=(0.01, 0.01),
    ),
)

source_galaxy = al.Galaxy(
    redshift=1.0,
    point_0=al.ps.Point(centre=(0.07, 0.07)),
)

tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

solver = al.PointSolver.for_grid(
    grid=grid, pixel_scale_precision=0.001, magnification_threshold=0.1
)

positions = solver.solve(
    tracer=tracer, source_plane_coordinate=source_galaxy.point_0.centre
)

# Position noise = 5 mas (HST PSF-centroiding precision), not the imaging pixel scale.
dataset = al.PointDataset(
    name="point_0",
    positions=positions,
    positions_noise_map=0.005,
)

al.output_to_json(
    obj=dataset,
    file_path=path.join(dataset_path, "point_dataset_positions_only.json"),
)

al.output_to_json(
    obj=tracer,
    file_path=path.join(dataset_path, "tracer.json"),
)

print(f"Saved point-source dataset with {len(positions)} images to {dataset_path}")
