"""
Jax Assertions: KNN Interpolator
================================

Exercises the JAX-based ``KNearestNeighbor`` mesh interpolator path through
``aa.Inversion``. The interpolator's ``mappings`` cached property calls
``get_interpolation_weights`` which uses ``jax.vmap`` and ``jax.numpy`` for
the per-query nearest-neighbor weighting (Wendland C4 kernel).

Asserts the expected ``pix_indexes_for_sub_slim_index`` and
``pix_weights_for_sub_slim_index`` values for the canonical
``knn_mapper_9_3x3`` fixture, plus the regularization matrix produced by both
``aa.reg.AdaptiveBrightness`` (default on the mapper) and
``aa.reg.AdaptSplit``.

Previously: ``test_autoarray/inversion/inversion/test_factory.py::test__inversion_imaging__via_mapper_knn``.
"""

import copy
import numpy as np
import numpy.testing as npt
import autoarray as aa

from autoarray import fixtures

"""
__Setup: KNN Mapper Fixture + Inversion__
"""
masked_imaging_7x7_no_blur = fixtures.make_masked_imaging_7x7_no_blur()
knn_mapper_9_3x3 = fixtures.make_knn_mapper_9_3x3()
regularization_adaptive_brightness_split = (
    fixtures.make_regularization_adaptive_brightness_split()
)

inversion = aa.Inversion(
    dataset=masked_imaging_7x7_no_blur,
    linear_obj_list=[knn_mapper_9_3x3],
)

"""
__pix_indexes_for_sub_slim_index (rows 0-2)__

Triggers the JAX nearest-neighbor weight computation.
"""
npt.assert_allclose(
    knn_mapper_9_3x3.pix_indexes_for_sub_slim_index[0, :],
    [1, 0, 4, 6, 2, 5, 3, 7, 8],
    rtol=1.0e-4,
)
npt.assert_allclose(
    knn_mapper_9_3x3.pix_indexes_for_sub_slim_index[1, :],
    [1, 0, 2, 4, 6, 3, 5, 7, 8],
    rtol=1.0e-4,
)
npt.assert_allclose(
    knn_mapper_9_3x3.pix_indexes_for_sub_slim_index[2, :],
    [1, 0, 4, 6, 2, 5, 3, 7, 8],
    rtol=1.0e-4,
)

"""
__pix_weights_for_sub_slim_index (rows 0-2)__

Wendland C4 weights from the JAX vmap'd interpolator.
"""
npt.assert_allclose(
    knn_mapper_9_3x3.pix_weights_for_sub_slim_index[0, :],
    [
        0.24139248,
        0.20182463,
        0.13465525,
        0.12882639,
        0.12169429,
        0.08682546,
        0.07062276,
        0.00982079,
        0.00433794,
    ],
    rtol=1.0e-4,
)
npt.assert_allclose(
    knn_mapper_9_3x3.pix_weights_for_sub_slim_index[1, :],
    [
        0.23255487,
        0.22727716,
        0.14466056,
        0.11643257,
        0.09868897,
        0.08878719,
        0.07744259,
        0.01010399,
        0.0040521,
    ],
    rtol=1.0e-4,
)
npt.assert_allclose(
    knn_mapper_9_3x3.pix_weights_for_sub_slim_index[2, :],
    [
        0.2334672,
        0.1785593,
        0.153417,
        0.15099354,
        0.11075057,
        0.09986048,
        0.06060822,
        0.00869774,
        0.00364596,
    ],
    rtol=1.0e-4,
)

"""
__Inversion Regularization (AdaptiveBrightness, default)__
"""
assert isinstance(inversion, aa.InversionImagingMapping)

npt.assert_allclose(
    inversion.regularization_matrix[0:3, 0],
    [4.00000001, -1.0, -1.0],
    rtol=1.0e-4,
)
npt.assert_allclose(
    inversion.regularization_matrix[0:3, 1],
    [-1.0, 3.00000001, 0.0],
    rtol=1.0e-4,
)
npt.assert_allclose(
    inversion.regularization_matrix[0:3, 2],
    [-1.0, 0.0, 4.00000001],
    rtol=1.0e-4,
)

npt.assert_allclose(
    inversion.log_det_curvature_reg_matrix_term, 10.417803331712355, rtol=1.0e-4
)
npt.assert_allclose(
    inversion.mapped_reconstructed_operated_data, np.ones(9), rtol=1.0e-4
)

"""
__Inversion Regularization (AdaptSplit)__

Swap the regularization to ``AdaptSplit`` and re-run the inversion.
"""
mapper = copy.copy(knn_mapper_9_3x3)
mapper.regularization = regularization_adaptive_brightness_split

inversion = aa.Inversion(
    dataset=masked_imaging_7x7_no_blur,
    linear_obj_list=[mapper],
)

npt.assert_allclose(
    inversion.regularization_matrix[0:3, 0],
    [22.47519068, -16.373819, 8.39424766],
    rtol=1.0e-4,
)
npt.assert_allclose(
    inversion.regularization_matrix[0:3, 1],
    [-16.373819, 112.1402519, -13.56808248],
    rtol=1.0e-4,
)
npt.assert_allclose(
    inversion.regularization_matrix[0:3, 2],
    [8.39424766, -13.56808248, 26.10743213],
    rtol=1.0e-4,
)

print("knn_interpolator: all assertions passed")
