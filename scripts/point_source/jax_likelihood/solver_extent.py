"""
PointSolver extent diagnostic and JAX likelihood regression.

Checks construction-time warnings without changing the image_plane.py likelihood
pin, then repeats both jit(fit_from) and vmapped evaluations without new logs.

__Env__

ENV: jax full_datasets
"""

import logging
import os
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import autofit as af
import autolens as al
from autofit.non_linear.fitness import Fitness

assert os.environ.get("PYAUTO_SMALL_DATASETS") != "1", "Extent parity needs full datasets"
assert os.environ.get("PYAUTO_DISABLE_JAX") != "1", "Extent parity needs JAX enabled"

# Same committed dataset and prior medians as image_plane.py.
dataset = al.from_json(
    file_path=Path("dataset/point_source/simple/point_dataset_positions_only.json")
)
mass = af.Model(al.mp.Isothermal)
mass.centre.centre_0 = af.UniformPrior(0.0, 0.02)
mass.centre.centre_1 = af.UniformPrior(0.0, 0.02)
mass.ell_comps.ell_comps_0 = af.UniformPrior(0.0, 0.02)
mass.ell_comps.ell_comps_1 = af.UniformPrior(0.0, 0.02)
mass.einstein_radius = af.UniformPrior(1.5, 1.8)
point = af.Model(al.ps.PointFlux)
point.centre.centre_0 = af.UniformPrior(0.06, 0.08)
point.centre.centre_1 = af.UniformPrior(0.06, 0.08)
galaxies = af.Collection(
    lens=af.Model(al.Galaxy, redshift=0.5, mass=mass),
    source=af.Model(al.Galaxy, redshift=1.0, point_0=point),
)
cosmology = af.Model(al.cosmo.FlatLambdaCDM)
cosmology.H0 = af.UniformPrior(0.0, 150.0)
model = af.Collection(galaxies=galaxies, cosmology=cosmology)
solver = al.PointSolver.for_grid(
    grid=al.Grid2D.uniform(shape_native=(100, 100), pixel_scales=0.2),
    pixel_scale_precision=0.001,
    magnification_threshold=0.1,
)


class Capture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record)


logger = logging.getLogger("autolens.point.model.analysis")
handler = Capture()
previous_level = logger.level
logger.setLevel(logging.INFO)
logger.addHandler(handler)
try:
    analysis = al.AnalysisPoint(
        dataset=dataset,
        solver=solver,
        fit_positions_cls=al.FitPositionsImagePairAll,
    )
    assert not any(r.levelno >= logging.WARNING for r in handler.records)
    handler.records.clear()
    small_solver = al.PointSolver.for_grid(
        grid=al.Grid2D.uniform(shape_native=(10, 10), pixel_scales=0.2),
        pixel_scale_precision=0.001,
    )
    al.AnalysisPoint(dataset=dataset, solver=small_solver)
    assert len(handler.records) == 1
    assert handler.records[0].levelno == logging.WARNING
    assert "does not guarantee image completeness" in handler.records[0].getMessage()
    handler.records.clear()

    fitness = Fitness(
        model=model,
        analysis=analysis,
        fom_is_log_likelihood=True,
        resample_figure_of_merit=-1.0e99,
    )
    parameters = jnp.array([model.physical_values_from_prior_medians])
    # No free cosmology for the jit fit round-trip, matching image_plane.py.
    instance = af.Collection(galaxies=galaxies).instance_from_prior_medians()
    fit_jit = jax.jit(analysis.fit_from)
    reference = (
        al.AnalysisPoint(
            dataset=dataset,
            solver=solver,
            fit_positions_cls=al.FitPositionsImagePairAll,
            use_jax=False,
        )
        .fit_from(instance)
        .log_likelihood
    )
    handler.records.clear()
    for _ in range(2):
        np.testing.assert_allclose(
            np.asarray(fitness._vmap(parameters)), -83.38049778, rtol=1e-4
        )
        np.testing.assert_allclose(
            float(fit_jit(instance).log_likelihood), float(reference), rtol=1e-4
        )
    assert (
        handler.records == []
    ), "Extent diagnostic repeated during likelihood evaluation"
finally:
    logger.removeHandler(handler)
    logger.setLevel(previous_level)

print("PASS: extent warning is construction-only; vmap pin and JIT parity unchanged")
