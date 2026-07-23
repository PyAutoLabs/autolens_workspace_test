"""
Test: batched simulate_substructure via vmap
============================================

Validates that batched_simulate_substructure (vmap over realizations)
produces identical results to calling simulate_substructure in a loop.

Ref: PyAutoLens#542, prompt 4.
"""
# ENV: jax full_datasets
# Tests the full simulate_substructure path at image_shape=(51,51);
# need JAX enabled and full-size datasets (the 15x15 cap raises).

import numpy as np
import jax
import jax.numpy as jnp

import autoarray as aa
import autogalaxy as ag
import autolens as al
from autolens.lens import substructure_util

"""
__Setup: shared configuration__
"""
grid = al.Grid2D.uniform(shape_native=(41, 41), pixel_scales=0.05)
grid_array = jnp.array(grid.array)
image_shape = (41, 41)

cosmology = ag.cosmo.Planck15()
plane_redshifts = [0.25, 0.5, 0.75, 1.0]
n_halos_per_plane = 3
max_n = 6
batch_size = 8

macro_galaxy = al.Galaxy(
    redshift=0.5,
    mass=al.mp.PowerLaw(
        centre=(0.0, 0.0),
        ell_comps=(0.05, -0.03),
        slope=2.2,
        einstein_radius=1.6,
    ),
    shear=al.mp.ExternalShear(gamma_1=0.01, gamma_2=-0.01),
)

source_galaxy = al.Galaxy(
    redshift=1.0,
    bulge=al.lp.SersicCore(
        centre=(0.02, -0.03),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=75.0),
        intensity=1.5,
        effective_radius=0.15,
        sersic_index=3.5,
        radius_break=0.025,
    ),
)

psf_sigma = 0.05
psf_shape = (11, 11)
y = np.arange(psf_shape[0]) - psf_shape[0] // 2
x = np.arange(psf_shape[1]) - psf_shape[1] // 2
yy, xx = np.meshgrid(y, x, indexing="ij")
psf_kernel = jnp.array(
    np.exp(-(yy**2 + xx**2) / (2 * (psf_sigma / grid.pixel_scales[0]) ** 2))
)
psf_kernel = psf_kernel / psf_kernel.sum()

scaling_matrix = substructure_util.precompute_scaling_matrix(
    plane_redshifts=plane_redshifts,
    cosmology=cosmology,
)


def lens_mass_fn(grid_raw, params):
    power_law = al.mp.PowerLaw(
        centre=(params[0], params[1]),
        ell_comps=(params[2], params[3]),
        slope=params[4],
        einstein_radius=params[5],
    )
    shear = al.mp.ExternalShear(gamma_1=params[6], gamma_2=params[7])
    galaxy = al.Galaxy(redshift=0.5, mass=power_law, shear=shear)
    g = aa.Grid2DIrregular(values=grid_raw, xp=jnp)
    return galaxy.deflections_yx_2d_from(grid=g, xp=jnp).array


lens_mass_params = jnp.array([0.0, 0.0, 0.05, -0.03, 2.2, 1.6, 0.01, -0.01])


def source_light_fn(grid_raw, params):
    bulge = al.lp.SersicCore(
        centre=(params[0], params[1]),
        ell_comps=(params[2], params[3]),
        intensity=params[4],
        effective_radius=params[5],
        sersic_index=params[6],
        radius_break=params[7],
    )
    galaxy = al.Galaxy(redshift=1.0, bulge=bulge)
    g = aa.Grid2DIrregular(values=grid_raw, xp=jnp)
    return galaxy.image_2d_from(grid=g, xp=jnp).array


source_light_params = jnp.array(
    [
        0.02,
        -0.03,
        0.05555555555555553,
        -0.0962250448649376,
        1.5,
        0.15,
        3.5,
        0.025,
    ]
)

lens_plane_mask = jnp.array(
    [1.0 if abs(z - 0.5) < 1e-6 else 0.0 for z in plane_redshifts]
)

"""
__Build B realizations with different random halo populations__
"""
all_realization_galaxies = []

for b in range(batch_size):
    np.random.seed(100 + b)
    galaxies = []

    for z in plane_redshifts[:3]:
        for _ in range(n_halos_per_plane):
            halo = ag.Galaxy(
                redshift=z,
                mass=ag.mp.NFWTruncatedSph(
                    centre=(
                        np.random.uniform(-0.5, 0.5),
                        np.random.uniform(-0.5, 0.5),
                    ),
                    kappa_s=np.random.uniform(0.001, 0.01),
                    scale_radius=np.random.uniform(0.5, 2.0),
                    truncation_radius=np.random.uniform(2.0, 5.0),
                ),
            )
            galaxies.append(halo)

        sheet = ag.Galaxy(
            redshift=z,
            mass_sheet=ag.mp.MassSheet(
                centre=(0.0, 0.0),
                kappa=-np.random.uniform(0.001, 0.005),
            ),
        )
        galaxies.append(sheet)

    all_realization_galaxies.append(galaxies)

halo_params_batch, halo_mask_batch, sheet_kappas_batch = (
    substructure_util.los_realizations_to_arrays(
        realization_galaxies=all_realization_galaxies,
        plane_redshifts=plane_redshifts,
        max_n=max_n,
        profile_cls=ag.mp.NFWTruncatedSph,
    )
)

keys = jax.random.split(jax.random.PRNGKey(42), batch_size)

lens_mass_params_batch = jnp.tile(lens_mass_params, (batch_size, 1))
source_light_params_batch = jnp.tile(source_light_params, (batch_size, 1))

"""
__Path A: sequential loop__
"""
images_sequential = []
for i in range(batch_size):
    img = substructure_util.simulate_substructure(
        grid=grid_array,
        image_shape=image_shape,
        halo_params=halo_params_batch[i],
        halo_mask=halo_mask_batch[i],
        scaling_matrix=scaling_matrix,
        lens_mass_fn=lens_mass_fn,
        lens_mass_params=lens_mass_params_batch[i],
        lens_plane_mask=lens_plane_mask,
        sheet_kappas=sheet_kappas_batch[i],
        source_light_fn=source_light_fn,
        source_light_params=source_light_params_batch[i],
        lens_light_fn=None,
        lens_light_params=None,
        lens_plane_idx=None,
        psf_kernel=psf_kernel,
        exposure_time=300.0,
        background_sky_level=0.1,
        prng_key=keys[i],
        halo_profile_cls=ag.mp.NFWTruncatedSph,
    )
    images_sequential.append(img)

"""
__Path B: batched via vmap__
"""
images_batched = substructure_util.batched_simulate_substructure(
    grid=grid_array,
    image_shape=image_shape,
    halo_params_batch=halo_params_batch,
    halo_mask_batch=halo_mask_batch,
    scaling_matrix=scaling_matrix,
    lens_mass_fn=lens_mass_fn,
    lens_mass_params_batch=lens_mass_params_batch,
    lens_plane_mask=lens_plane_mask,
    sheet_kappas_batch=sheet_kappas_batch,
    source_light_fn=source_light_fn,
    source_light_params_batch=source_light_params_batch,
    lens_light_fn=None,
    lens_light_params_batch=None,
    lens_plane_idx=None,
    psf_kernel=psf_kernel,
    exposure_time=300.0,
    background_sky_level=0.1,
    prng_keys=keys,
    halo_profile_cls=ag.mp.NFWTruncatedSph,
)

"""
__Compare batch vs sequential__
"""
for i in range(batch_size):
    np.testing.assert_allclose(
        np.array(images_batched[i]),
        np.array(images_sequential[i]),
        atol=1e-5,
        err_msg=f"Batch element {i} does not match sequential result",
    )
print(f"PASS: All {batch_size} batch elements match sequential results")

"""
__JIT compilation__
"""
jitted_batch = jax.jit(
    lambda hp, hm, sk, lmp, slp, k: substructure_util.batched_simulate_substructure(
        grid=grid_array,
        image_shape=image_shape,
        halo_params_batch=hp,
        halo_mask_batch=hm,
        scaling_matrix=scaling_matrix,
        lens_mass_fn=lens_mass_fn,
        lens_mass_params_batch=lmp,
        lens_plane_mask=lens_plane_mask,
        sheet_kappas_batch=sk,
        source_light_fn=source_light_fn,
        source_light_params_batch=slp,
        lens_light_fn=None,
        lens_light_params_batch=None,
        lens_plane_idx=None,
        psf_kernel=psf_kernel,
        exposure_time=300.0,
        background_sky_level=0.1,
        prng_keys=k,
        halo_profile_cls=ag.mp.NFWTruncatedSph,
    )
)

images_jit = jitted_batch(
    halo_params_batch,
    halo_mask_batch,
    sheet_kappas_batch,
    lens_mass_params_batch,
    source_light_params_batch,
    keys,
)

np.testing.assert_allclose(
    np.array(images_jit),
    np.array(images_batched),
    atol=1e-5,
    err_msg="JIT batched result doesn't match eager batched result",
)
print("PASS: jax.jit(batched_simulate_substructure) compiles and matches")

"""
__Different realizations produce different images__
"""
assert not jnp.allclose(
    images_batched[0], images_batched[1]
), "Different realizations produced identical images"
print("PASS: Different realizations produce different images")

print(f"\nAll tests passed. Batch shape: {images_batched.shape}")
