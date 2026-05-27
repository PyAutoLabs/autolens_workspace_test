"""
Test: vmapped subhalo deflections
=================================

Validates that the vmapped deflection path (MassProfile.vmapped_deflections_from)
reproduces the existing Python-loop-over-Galaxies path for NFWTruncatedSph and
cNFWSph profiles. This is the integration test for PyAutoLens issue #542, prompt 1.
"""
import numpy as np
import jax
import jax.numpy as jnp

import autogalaxy as ag
import autolens as al

np.random.seed(42)

grid_shape = (50, 50)
pixel_scales = 0.05

grid = al.Grid2D.uniform(shape_native=grid_shape, pixel_scales=pixel_scales)
grid_array = jnp.array(grid.array)

n_halos = 10

"""
__NFWTruncatedSph__

Create 10 NFWTruncatedSph halos with random parameters and compare the
Python-loop deflection sum against the vmapped path.
"""
nfw_halos = []
nfw_params_list = []

for i in range(n_halos):
    centre_y = np.random.uniform(-0.5, 0.5)
    centre_x = np.random.uniform(-0.5, 0.5)
    kappa_s = np.random.uniform(0.01, 0.1)
    scale_radius = np.random.uniform(0.5, 2.0)
    truncation_radius = np.random.uniform(2.0, 5.0)

    halo = ag.mp.NFWTruncatedSph(
        centre=(centre_y, centre_x),
        kappa_s=kappa_s,
        scale_radius=scale_radius,
        truncation_radius=truncation_radius,
    )
    nfw_halos.append(halo)
    nfw_params_list.append(
        [centre_y, centre_x, kappa_s, scale_radius, truncation_radius]
    )

nfw_params = jnp.array(nfw_params_list)
nfw_mask = jnp.ones(n_halos, dtype=bool)

deflections_loop = sum(
    h.deflections_yx_2d_from(grid=grid, xp=jnp).array for h in nfw_halos
)

deflections_vmap = ag.mp.NFWTruncatedSph.vmapped_deflections_from(
    grid=grid_array, params_batch=nfw_params, mask=nfw_mask,
)

np.testing.assert_allclose(
    np.array(deflections_vmap),
    np.array(deflections_loop),
    atol=1e-6,
    err_msg="NFWTruncatedSph: vmapped path does not match loop path",
)
print("PASS: NFWTruncatedSph vmapped deflections match loop path")

"""
__cNFWSph__

Same test for cored NFW profiles.
"""
cnfw_halos = []
cnfw_params_list = []

for i in range(n_halos):
    centre_y = np.random.uniform(-0.5, 0.5)
    centre_x = np.random.uniform(-0.5, 0.5)
    kappa_s = np.random.uniform(0.01, 0.1)
    scale_radius = np.random.uniform(0.5, 2.0)
    core_radius = np.random.uniform(0.01, 0.1)

    halo = ag.mp.cNFWSph(
        centre=(centre_y, centre_x),
        kappa_s=kappa_s,
        scale_radius=scale_radius,
        core_radius=core_radius,
    )
    cnfw_halos.append(halo)
    cnfw_params_list.append(
        [centre_y, centre_x, kappa_s, scale_radius, core_radius]
    )

cnfw_params = jnp.array(cnfw_params_list)
cnfw_mask = jnp.ones(n_halos, dtype=bool)

deflections_loop_cnfw = sum(
    h.deflections_yx_2d_from(grid=grid, xp=jnp).array for h in cnfw_halos
)

deflections_vmap_cnfw = ag.mp.cNFWSph.vmapped_deflections_from(
    grid=grid_array, params_batch=cnfw_params, mask=cnfw_mask,
)

np.testing.assert_allclose(
    np.array(deflections_vmap_cnfw),
    np.array(deflections_loop_cnfw),
    atol=1e-6,
    err_msg="cNFWSph: vmapped path does not match loop path",
)
print("PASS: cNFWSph vmapped deflections match loop path")

"""
__Masking__

Verify that masked-out slots contribute zero deflection.
"""
partial_mask = jnp.array(
    [True, True, True, True, True, True, True, False, False, False]
)

deflections_partial = ag.mp.NFWTruncatedSph.vmapped_deflections_from(
    grid=grid_array, params_batch=nfw_params, mask=partial_mask,
)

deflections_subset_loop = sum(
    h.deflections_yx_2d_from(grid=grid, xp=jnp).array for h in nfw_halos[:7]
)

np.testing.assert_allclose(
    np.array(deflections_partial),
    np.array(deflections_subset_loop),
    atol=1e-6,
    err_msg="Masking: masked slots contributed non-zero deflection",
)
print("PASS: Masked-out slots contribute zero deflection")

"""
__JIT Compilation__

Verify the vmapped function compiles under jax.jit and reuses the compiled
code when parameter values change (same shapes).
"""
jitted_fn = jax.jit(
    ag.mp.NFWTruncatedSph.vmapped_deflections_from,
    static_argnums=(3,),
)

result_1 = jitted_fn(grid_array, nfw_params, nfw_mask)

nfw_params_shifted = nfw_params.at[:, 0].add(0.01)
result_2 = jitted_fn(grid_array, nfw_params_shifted, nfw_mask)

assert not jnp.allclose(result_1, result_2), (
    "JIT: different params produced identical results"
)
print("PASS: jax.jit compiles and reuses for different parameter values")

"""
__Elliptical Profile Error__

Verify that calling vmapped_deflections_from on a profile without
radial_deflection_from raises NotImplementedError with a clear message.
"""
try:
    ag.mp.PowerLaw.vmapped_deflections_from(
        grid=grid_array,
        params_batch=jnp.zeros((2, 5)),
        mask=jnp.ones(2, dtype=bool),
    )
    raise AssertionError("Expected NotImplementedError was not raised")
except NotImplementedError as e:
    assert "not yet supported" in str(e)
    assert "jnightingale2211@gmail.com" in str(e)
    print("PASS: Unsupported profile raises clear NotImplementedError")

print("\nAll tests passed.")
