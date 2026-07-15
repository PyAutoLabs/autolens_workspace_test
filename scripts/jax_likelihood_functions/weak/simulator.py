"""
Simulator for the weak-lensing JAX parity scripts.

Writes a seeded, noise-controlled `WeakDataset` to `dataset/weak/simple/dataset.json` so the
`shear.py` vmap-parity regression constant is stable across runs.
"""
from pathlib import Path

import autoarray as aa
import autolens as al

import numpy as np

dataset_path = Path("dataset") / "weak" / "simple"

grid = aa.Grid2DIrregular(values=np.random.default_rng(1).uniform(-3.0, 3.0, (100, 2)))

tracer = al.Tracer(
    galaxies=[
        al.Galaxy(
            redshift=0.5,
            mass=al.mp.Isothermal(
                centre=(0.0, 0.0),
                ell_comps=(0.0, 0.05),
                einstein_radius=1.6,
            ),
        ),
        al.Galaxy(redshift=1.0),
    ]
)

dataset = al.SimulatorShearYX(noise_sigma=0.3, seed=1).via_tracer_from(
    tracer=tracer, grid=grid, name="simple"
)

dataset_path.mkdir(parents=True, exist_ok=True)
al.output_to_json(obj=dataset, file_path=dataset_path / "dataset.json")
al.output_to_json(obj=tracer, file_path=dataset_path / "tracer.json")

print(f"Wrote weak parity dataset to {dataset_path}")
