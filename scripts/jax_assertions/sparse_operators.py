"""
Jax Assertions: Sparse Operators
================================

Cross-implementation parity checks for the JAX-accelerated sparse-operator
fast paths in ``autoarray.inversion``:

- ``data_vector_via_psf_weighted_data_from`` (uses ``jax.ops.segment_sum``) is
  compared against the dense ``data_vector_via_blurred_mapping_matrix_from``
  reference.
- ``ImagingSparseOperator.from_noise_map_and_psf`` (FFT-based curvature
  diagonal in JAX) is compared against the dense
  ``curvature_matrix_via_mapping_matrix_from`` reference.
- ``InterferometerSparseOperator.from_nufft_precision_operator`` (FFT-based
  curvature in JAX) is compared against
  ``curvature_matrix_diag_via_psf_weighted_noise_from``.

These previously lived as pytest tests in ``test_autoarray/`` but pulled the
``jax`` import into the library unit-test run. They are now executable
scripts so unit tests stay numpy-only.
"""

import numpy as np
import numpy.testing as npt
import autoarray as aa

"""
__Data Vector via PSF-Weighted Data (segment_sum) vs Blurred Mapping Matrix__

For two sub-grid sizes, build a mapper, compute the data vector two ways
(dense blurred mapping matrix, and sparse-triplet path through
``data_vector_via_psf_weighted_data_from`` which uses
``jax.ops.segment_sum``), and assert agreement to 1e-4.
"""
mask = aa.Mask2D.circular(shape_native=(51, 51), pixel_scales=0.1, radius=2.0)

image = np.random.uniform(size=mask.shape_native)
image = aa.Array2D(values=image, mask=mask)

noise_map = np.random.uniform(size=mask.shape_native)
noise_map = aa.Array2D(values=noise_map, mask=mask)

convolver = aa.Convolver.from_gaussian(
    shape_native=(7, 7), pixel_scales=mask.pixel_scales, sigma=1.0, normalize=True
)

psf = convolver

mesh = aa.mesh.RectangularUniform(shape=(20, 20))

for sub_size in range(1, 3):

    grid = aa.Grid2D.from_mask(mask=mask, over_sample_size=sub_size)

    interpolator = mesh.interpolator_from(
        source_plane_data_grid=grid, source_plane_mesh_grid=None
    )

    mapper = aa.Mapper(interpolator=interpolator)

    mapping_matrix = mapper.mapping_matrix

    blurred_mapping_matrix = psf.convolved_mapping_matrix_from(
        mapping_matrix=mapping_matrix, mask=mask
    )

    data_vector = aa.util.inversion_imaging.data_vector_via_blurred_mapping_matrix_from(
        blurred_mapping_matrix=blurred_mapping_matrix,
        image=image,
        noise_map=noise_map,
    )

    rows, cols, vals = aa.util.mapper.sparse_triplets_from(
        pix_indexes_for_sub=mapper.pix_indexes_for_sub_slim_index,
        pix_weights_for_sub=mapper.pix_weights_for_sub_slim_index,
        slim_index_for_sub=mapper.slim_index_for_sub_slim_index,
        fft_index_for_masked_pixel=mask.fft_index_for_masked_pixel,
        sub_fraction_slim=mapper.over_sampler.sub_fraction.array,
    )

    weight_map = image.array / (noise_map.array**2)
    weight_map = aa.Array2D(values=weight_map, mask=noise_map.mask)

    psf_weighted_data = aa.util.inversion_imaging.psf_weighted_data_from(
        weight_map_native=weight_map.native.array,
        kernel_native=convolver.kernel.native.array,
        native_index_for_slim_index=mask.derive_indexes.native_for_slim.astype("int"),
    )

    data_vector_via_psf_weighted_noise = (
        aa.util.inversion_imaging.data_vector_via_psf_weighted_data_from(
            psf_weighted_data=psf_weighted_data,
            rows=rows,
            cols=cols,
            vals=vals,
            S=mesh.pixels,
        )
    )

    npt.assert_allclose(
        np.asarray(data_vector_via_psf_weighted_noise),
        np.asarray(data_vector),
        rtol=1.0e-4,
    )

"""
__ImagingSparseOperator vs Curvature Matrix via Mapping Matrix__

Build an ``ImagingSparseOperator`` (FFT/JAX-based curvature) and compare its
diagonal curvature against the dense
``curvature_matrix_via_mapping_matrix_from`` reference for an
adaptive-density rectangular mesh, to abs tol 1e-4.
"""
mask = aa.Mask2D.circular(shape_native=(21, 21), pixel_scales=0.1, radius=0.8)

noise_map = np.random.uniform(size=mask.shape_native)
noise_map = aa.Array2D(values=noise_map, mask=mask)

kernel = aa.Convolver.from_gaussian(
    shape_native=(5, 5), pixel_scales=mask.pixel_scales, sigma=1.0, normalize=True
)

psf = kernel

sparse_operator = aa.ImagingSparseOperator.from_noise_map_and_psf(
    data=noise_map,
    noise_map=noise_map,
    psf=psf.kernel.native,
)

mesh = aa.mesh.RectangularAdaptDensity(shape=(8, 8))

interpolator = mesh.interpolator_from(
    source_plane_data_grid=mask.derive_grid.unmasked,
    source_plane_mesh_grid=None,
)

mapper = aa.Mapper(interpolator=interpolator)

mapping_matrix = mapper.mapping_matrix

rows, cols, vals = aa.util.mapper.sparse_triplets_from(
    pix_indexes_for_sub=mapper.pix_indexes_for_sub_slim_index,
    pix_weights_for_sub=mapper.pix_weights_for_sub_slim_index,
    slim_index_for_sub=mapper.slim_index_for_sub_slim_index,
    fft_index_for_masked_pixel=mask.fft_index_for_masked_pixel,
    sub_fraction_slim=mapper.over_sampler.sub_fraction.array,
    return_rows_slim=False,
)

curvature_matrix_via_sparse_operator = sparse_operator.curvature_matrix_diag_from(
    rows,
    cols,
    vals,
    S=mesh.shape[0] * mesh.shape[1],
)

curvature_matrix_via_sparse_operator = (
    aa.util.inversion_imaging.curvature_matrix_mirrored_from(
        curvature_matrix=curvature_matrix_via_sparse_operator,
    )
)

blurred_mapping_matrix = psf.convolved_mapping_matrix_from(
    mapping_matrix=mapping_matrix, mask=mask
)

curvature_matrix = aa.util.inversion.curvature_matrix_via_mapping_matrix_from(
    mapping_matrix=blurred_mapping_matrix,
    noise_map=noise_map,
)

npt.assert_allclose(
    np.asarray(curvature_matrix_via_sparse_operator),
    np.asarray(curvature_matrix),
    atol=1.0e-4,
)

"""
__InterferometerSparseOperator vs PSF-Weighted Noise Curvature__

Build an ``InterferometerSparseOperator`` from the NUFFT precision operator
and compare its sparse-operator curvature matrix against the
``curvature_matrix_diag_via_psf_weighted_noise_from`` reference, to rel
tolerance 1e-4.
"""
noise_map = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
uv_wavelengths = np.array(
    [[0.0001, 2.0, 3000.0, 50000.0, 200000.0], [3000.0, 2.0, 0.0001, 10.0, 5000.0]]
)

grid = aa.Grid2D.uniform(shape_native=(3, 3), pixel_scales=0.0005)

mapping_matrix = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
    ]
)

nufft_precision_operator = (
    aa.util.inversion_interferometer.nufft_precision_operator_from(
        noise_map_real=noise_map,
        uv_wavelengths=uv_wavelengths,
        shape_masked_pixels_2d=(3, 3),
        grid_radians_2d=np.array(grid.native),
    )
)

native_index_for_slim_index = np.array(
    [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2]]
)

psf_weighted_noise = (
    aa.util.inversion_interferometer.nufft_weighted_noise_via_sparse_operator_from(
        translation_invariant_kernel=nufft_precision_operator,
        native_index_for_slim_index=native_index_for_slim_index,
    )
)

curvature_matrix_via_nufft_weighted_noise = (
    aa.util.inversion.curvature_matrix_diag_via_psf_weighted_noise_from(
        psf_weighted_noise=psf_weighted_noise, mapping_matrix=mapping_matrix
    )
)

pix_indexes_for_sub_slim_index = np.array([[0], [2], [1], [1], [2], [2], [0], [2], [0]])

pix_weights_for_sub_slim_index = np.ones(shape=(9, 1))

sparse_operator = aa.InterferometerSparseOperator.from_nufft_precision_operator(
    nufft_precision_operator=nufft_precision_operator,
    dirty_image=None,
)

curvature_matrix_via_preload = (
    sparse_operator.curvature_matrix_via_sparse_operator_from(
        pix_indexes_for_sub_slim_index=pix_indexes_for_sub_slim_index,
        pix_weights_for_sub_slim_index=pix_weights_for_sub_slim_index,
        fft_index_for_masked_pixel=grid.mask.fft_index_for_masked_pixel,
        pix_pixels=3,
    )
)

npt.assert_allclose(
    np.asarray(curvature_matrix_via_nufft_weighted_noise),
    np.asarray(curvature_matrix_via_preload),
    rtol=1.0e-4,
)

print("sparse_operators: all assertions passed")
