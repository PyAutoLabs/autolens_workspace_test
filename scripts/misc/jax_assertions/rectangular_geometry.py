"""Adapt-image cell edges and areas retain their raw-array JAX JIT contract."""

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import autoarray as aa

jax.config.update("jax_enable_x64", True)


grid = aa.Grid2D.uniform(shape_native=(5, 5), pixel_scales=0.4, over_sample_size=2)


def geometry_from(values, xp):
    adapt_data = aa.Array2D(values=values.ravel(), mask=grid.mask, xp=xp)
    mesh = aa.mesh.RectangularBilinearAdaptImage(
        shape=(6, 6), respect_small_datasets=False
    )
    mapper = aa.Mapper(
        interpolator=mesh.interpolator_from(
            source_plane_data_grid=grid,
            source_plane_mesh_grid=None,
            adapt_data=adapt_data,
            xp=xp,
        )
    )
    geometry = mapper.mesh_geometry
    return geometry.edges_transformed, geometry.areas_transformed


values = np.arange(1.0, 26.0).reshape(5, 5) ** 2
numpy_result = geometry_from(values, np)
jax_result = geometry_from(jnp.asarray(values), jnp)
compiled_result = jax.jit(lambda image: geometry_from(image, jnp))(jnp.asarray(values))
for result, array_type in (
    (numpy_result, np.ndarray),
    (jax_result, jax.Array),
    (compiled_result, jax.Array),
):
    edges, areas = result
    assert isinstance(edges, array_type)
    assert isinstance(areas, array_type)
    assert edges.shape == (7, 2)
    assert areas.shape == (36,)
    npt.assert_allclose(edges, numpy_result[0], rtol=1e-10, atol=1e-10)
    npt.assert_allclose(areas, numpy_result[1], rtol=1e-10, atol=1e-10)
    expected = np.outer(
        -np.diff(np.asarray(edges)[:, 0]), np.diff(np.asarray(edges)[:, 1])
    ).ravel()
    npt.assert_allclose(areas, expected, rtol=1e-10, atol=1e-10)
    assert np.all(np.isfinite(areas))
    assert np.all(np.asarray(areas) >= 0.0)
print("PASS: adapt-image rectangular geometry NumPy/JAX/JIT edges and areas agree")
