"""
Visualization JAX Pilot: Point Source Analysis
===============================================

Pilot for the JAX-backed visualization path on ``PointDataset``.

Goal
----
Run ``VisualizerPoint.visualize`` with ``use_jax=True`` and
``use_jax_for_visualization=True`` on ``AnalysisPoint``. The point
visualizer dispatches through ``analysis.fit_for_visualization``, which
lazily wraps ``fit_from`` in ``jax.jit``. To trace across that boundary the
model and fit return type must be JAX pytrees, so this script enables pytree
registration before constructing the model.

Scope
-----
- ``Isothermal`` lens mass + ``PointFlux`` source (image-plane chi-squared).
- Calls ``VisualizerPoint.visualize`` only (not ``visualize_before_fit``).
- Re-uses the ``simple/point_dataset_positions_only.json`` dataset.
- No ``try/except`` wrapper — failure surfaces immediately.
"""

import shutil
from pathlib import Path
from types import SimpleNamespace

import autofit as af
import autolens as al
from autofit.jax.pytrees import enable_pytrees, register_model
from autolens.point.model.visualizer import VisualizerPoint

enable_pytrees()


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

register_model(model)


"""
__Analysis__

``use_jax=True`` turns on the JAX ``_xp`` path;
``use_jax_for_visualization=True`` tells the visualization path to wrap
``fit_from`` in ``jax.jit`` via ``Analysis.fit_for_visualization``.
``title_prefix`` is passed through via PR #506's **kwargs fix.
"""
analysis = al.AnalysisPoint(
    dataset=dataset,
    solver=solver,
    fit_positions_cls=al.FitPositionsImagePairAll,
    use_jax=True,
    use_jax_for_visualization=True,
    title_prefix="JAX_PILOT",
)


"""
__Paths__
"""
image_path = Path("scripts") / "point_source" / "images" / "visualization_jax"
if image_path.exists():
    shutil.rmtree(image_path)
image_path.mkdir(parents=True)
output_path = image_path / "output"
output_path.mkdir(parents=True)
paths = SimpleNamespace(image_path=image_path, output_path=output_path)


"""
__Run visualize on the JAX-backed fit__
"""
instance = model.instance_from_prior_medians()

print("Running VisualizerPoint.visualize with use_jax_for_visualization=True ...")
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
print("PILOT SUCCEEDED — JAX-backed point-source visualization produced fit.png.")
