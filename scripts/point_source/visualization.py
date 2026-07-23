"""
Visualization NumPy: Point Source Analysis
==========================================

Tests that ``VisualizerPoint.visualize`` runs end-to-end on a
``PointDataset`` using the NumPy (non-JAX) code path and that ``fit.png``
lands on disk.

Uses the ``simple/point_dataset_positions_only.json`` dataset (auto-simulated
if missing) with an ``Isothermal`` lens mass and ``PointFlux`` source — the
same model that is proven to JIT end-to-end in
``scripts/jax_likelihood_functions/point_source/image_plane.py``.

No ``try/except`` — any failure in the visualizer surfaces immediately.
"""

"""
__Env__

Test-harness configuration (PyAutoHands docs/env_profile_redesign.md §10).
Asserts fit.png lands on disk (needs real plots); uses a JSON point dataset
unaffected by SMALL_DATASETS.

ENV: real_plots
"""

import shutil
from pathlib import Path
from types import SimpleNamespace

import autofit as af
import autolens as al
from autolens.point.model.visualizer import VisualizerPoint


"""
__Dataset__
"""
dataset_path = Path("dataset") / "point_source" / "simple"

if al.util.dataset.should_simulate(str(dataset_path)):
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "scripts/jax_likelihood_functions/point_source/simulator.py"],
        check=True,
    )

dataset = al.from_json(
    file_path=dataset_path / "point_dataset_positions_only.json",
)


"""
__Point Solver__
"""
grid = al.Grid2D.uniform(shape_native=(100, 100), pixel_scales=0.2)

solver = al.PointSolver.for_grid(
    grid=grid, pixel_scale_precision=0.001, magnification_threshold=0.1
)


"""
__Model__

Tight priors centred on the true values so the prior-median instance
produces a sensible lens configuration (multiple images exist).
No free cosmology — cosmology distance caching breaks JIT.
"""
mass = af.Model(al.mp.Isothermal)
mass.centre.centre_0 = af.UniformPrior(lower_limit=0.0, upper_limit=0.02)
mass.centre.centre_1 = af.UniformPrior(lower_limit=0.0, upper_limit=0.02)
mass.ell_comps.ell_comps_0 = af.UniformPrior(lower_limit=0.0, upper_limit=0.02)
mass.ell_comps.ell_comps_1 = af.UniformPrior(lower_limit=0.0, upper_limit=0.02)
mass.einstein_radius = af.UniformPrior(lower_limit=1.5, upper_limit=1.8)

lens = af.Model(al.Galaxy, redshift=0.5, mass=mass)

point_0 = af.Model(al.ps.PointFlux)
point_0.centre.centre_0 = af.UniformPrior(lower_limit=0.06, upper_limit=0.08)
point_0.centre.centre_1 = af.UniformPrior(lower_limit=0.06, upper_limit=0.08)

source = af.Model(al.Galaxy, redshift=1.0, point_0=point_0)

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))


"""
__Analysis__

Explicit NumPy path (use_jax=False).
"""
analysis = al.AnalysisPoint(
    dataset=dataset,
    solver=solver,
    fit_positions_cls=al.FitPositionsImagePairAll,
    use_jax=False,
)


"""
__Paths__
"""
image_path = Path("scripts") / "point_source" / "images" / "visualization"
if image_path.exists():
    shutil.rmtree(image_path)
image_path.mkdir(parents=True)
output_path = image_path / "output"
output_path.mkdir(parents=True)
paths = SimpleNamespace(image_path=image_path, output_path=output_path)


"""
__Visualize__
"""
instance = model.instance_from_prior_medians()

print("Running VisualizerPoint.visualize (NumPy) ...")
VisualizerPoint.visualize(
    analysis=analysis,
    paths=paths,
    instance=instance,
    during_analysis=False,
)

print("Files in image_path:", list(image_path.iterdir()))
assert (
    image_path / "fit.png"
).exists(), f"fit.png was not produced. Files present: {list(image_path.iterdir())}"
print("NumPy point-source visualization produced fit.png.")
