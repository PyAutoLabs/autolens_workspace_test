"""
Jax Assertions: FitImaging Sparse Operator
==========================================

Cross-implementation parity check for the JAX-accelerated
``ImagingSparseOperator`` path inside a multi-plane ``FitImaging`` fit.

A complex tracer (3 lens planes with linear Sersic light + Isothermal mass,
2 source planes with ``RectangularUniform`` pixelizations) is fitted twice:

- via the standard mapping-matrix path, and
- via ``masked_dataset.apply_sparse_operator()`` (FFT-based JAX path).

The two fits must agree on the inversion ``curvature_matrix`` and
``regularization_matrix`` to ``rtol=1e-4``.

Previously: ``test_autolens/imaging/test_simulate_and_fit_imaging.py``
function ``test__simulate_imaging_data_and_fit__complex_fit_compare_mapping_matrix_sparse_operator``.
Moved here so PyAutoLens unit tests stay numpy-only.
"""

import numpy as np
import numpy.testing as npt

import autolens as al


grid = al.Grid2D.uniform(shape_native=(15, 15), pixel_scales=0.1)

psf = al.Convolver.from_gaussian(
    shape_native=(3, 3), pixel_scales=0.2, sigma=0.75, normalize=True
)

lens_0 = al.Galaxy(
    redshift=0.1,
    light=al.lp.Sersic(centre=(0.1, 0.1)),
    mass=al.mp.Isothermal(centre=(0.1, 0.1), einstein_radius=0.2),
)

lens_1 = al.Galaxy(
    redshift=0.2,
    light=al.lp.Sersic(centre=(0.2, 0.2)),
    mass=al.mp.Isothermal(centre=(0.1, 0.1), einstein_radius=0.2),
)

lens_2 = al.Galaxy(
    redshift=0.3,
    light=al.lp.Sersic(centre=(0.3, 0.3)),
    mass=al.mp.Isothermal(centre=(0.1, 0.1), einstein_radius=0.2),
)

source_0 = al.Galaxy(redshift=0.4, bulge=al.lp.Sersic(centre=(0.3, 0.3)))
source_1 = al.Galaxy(redshift=0.5, bulge=al.lp.Sersic(centre=(0.3, 0.3)))
tracer = al.Tracer(galaxies=[lens_0, lens_1, lens_2, source_0, source_1])

dataset = al.SimulatorImaging(
    exposure_time=300.0, psf=psf, add_poisson_noise_to_data=False
)

dataset = dataset.via_tracer_from(tracer=tracer, grid=grid)
dataset.sub_size = 2
dataset.noise_map = al.Array2D.ones(
    shape_native=dataset.data.shape_native, pixel_scales=0.2
)
mask = al.Mask2D.circular(
    shape_native=dataset.data.shape_native, pixel_scales=0.2, radius=0.8
)

masked_dataset = dataset.apply_mask(mask=mask)

lens_0 = al.Galaxy(
    redshift=0.1,
    light=al.lp_linear.Sersic(centre=(0.1, 0.1)),
    mass=al.mp.Isothermal(centre=(0.1, 0.1), einstein_radius=0.2),
)

lens_1 = al.Galaxy(
    redshift=0.2,
    light=al.lp_linear.Sersic(centre=(0.2, 0.2)),
    mass=al.mp.Isothermal(centre=(0.1, 0.1), einstein_radius=0.2),
)

lens_2 = al.Galaxy(
    redshift=0.3,
    light=al.lp_linear.Sersic(centre=(0.3, 0.3)),
    mass=al.mp.Isothermal(centre=(0.1, 0.1), einstein_radius=0.2),
)

pixelization = al.Pixelization(
    mesh=al.mesh.RectangularUniform(shape=(3, 3)),
    regularization=al.reg.Constant(coefficient=1.0),
)

source_0 = al.Galaxy(redshift=0.4, pixelization=pixelization)

pixelization = al.Pixelization(
    mesh=al.mesh.RectangularUniform(shape=(3, 3)),
    regularization=al.reg.Constant(coefficient=1.0),
)

source_1 = al.Galaxy(redshift=0.5, pixelization=pixelization)

tracer = al.Tracer(galaxies=[lens_0, lens_1, lens_2, source_0, source_1])

fit_mapping = al.FitImaging(
    dataset=masked_dataset,
    tracer=tracer,
)

masked_dataset_sparse_operator = masked_dataset.apply_sparse_operator()

fit_sparse_operator = al.FitImaging(
    dataset=masked_dataset_sparse_operator,
    tracer=tracer,
)

npt.assert_allclose(
    np.asarray(fit_mapping.inversion.curvature_matrix),
    np.asarray(fit_sparse_operator.inversion.curvature_matrix),
    rtol=1.0e-4,
)

npt.assert_allclose(
    np.asarray(fit_mapping.inversion.regularization_matrix),
    np.asarray(fit_sparse_operator.inversion.regularization_matrix),
    rtol=1.0e-4,
)

print("fit_imaging_sparse_operator: all assertions passed")
