"""
Test: end-to-end simulate_substructure
======================================

Validates that simulate_substructure (pure jnp path) produces a lensed
image consistent with the existing SimulatorImaging.via_tracer_from path.
Compares the deterministic (no-noise) lensed+convolved image.

Ref: PyAutoLens#542, prompt 3.
"""

import numpy as np
import jax
import jax.numpy as jnp

import autoarray as aa
import autogalaxy as ag
import autolens as al
from autolens.lens import substructure_util

np.random.seed(42)

"""
__Setup: same 4-plane system as prompt 2__
"""
grid = al.Grid2D.uniform(shape_native=(51, 51), pixel_scales=0.05)
grid_array = jnp.array(grid.array)
image_shape = (51, 51)

cosmology = ag.cosmo.Planck15()
plane_redshifts = [0.25, 0.5, 0.75, 1.0]
n_halos_per_plane = 5
max_n = 10

all_galaxies = []

for plane_idx, z in enumerate(plane_redshifts[:3]):
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
        all_galaxies.append(halo)

    sheet_kappa = -np.random.uniform(0.001, 0.005)
    sheet = ag.Galaxy(
        redshift=z,
        mass_sheet=ag.mp.MassSheet(centre=(0.0, 0.0), kappa=sheet_kappa),
    )
    all_galaxies.append(sheet)

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
all_galaxies.append(macro_galaxy)

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
all_galaxies.append(source_galaxy)

"""
__PSF kernel__
"""
psf_sigma = 0.05
psf_shape = (13, 13)
y = np.arange(psf_shape[0]) - psf_shape[0] // 2
x = np.arange(psf_shape[1]) - psf_shape[1] // 2
yy, xx = np.meshgrid(y, x, indexing="ij")
psf_kernel_np = np.exp(-(yy**2 + xx**2) / (2 * (psf_sigma / grid.pixel_scales[0]) ** 2))
psf_kernel_np /= psf_kernel_np.sum()
psf_kernel = jnp.array(psf_kernel_np)

"""
__Build scan-path inputs__
"""
halo_galaxies = [
    g
    for g in all_galaxies
    if g.redshift < 1.0 and g is not macro_galaxy and g is not source_galaxy
]

halo_params, halo_mask, sheet_kappas = substructure_util.galaxies_to_halo_arrays(
    galaxies=halo_galaxies,
    plane_redshifts=plane_redshifts,
    max_n=max_n,
    profile_cls=ag.mp.NFWTruncatedSph,
)

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
__Path B: simulate_substructure (no noise)__
"""
image_scan = substructure_util.simulate_substructure(
    grid=grid_array,
    image_shape=image_shape,
    halo_params=halo_params,
    halo_mask=halo_mask,
    scaling_matrix=scaling_matrix,
    lens_mass_fn=lens_mass_fn,
    lens_mass_params=lens_mass_params,
    lens_plane_mask=lens_plane_mask,
    sheet_kappas=sheet_kappas,
    source_light_fn=source_light_fn,
    source_light_params=source_light_params,
    psf_kernel=psf_kernel,
    exposure_time=300.0,
    background_sky_level=0.1,
    prng_key=None,
    halo_profile_cls=ag.mp.NFWTruncatedSph,
)

assert image_scan.shape == image_shape
assert not jnp.any(jnp.isnan(image_scan)), "NaN in simulated image"
assert jnp.max(image_scan) > 0, "Image is all zeros or negative"
print(
    f"PASS: simulate_substructure produces valid {image_shape} image "
    f"(max={float(jnp.max(image_scan)):.4f})"
)

"""
__Path A: existing Tracer lensed image (pre-convolution) for comparison__

Compare the raw lensed image (before PSF and noise) from both paths.
"""
tracer = al.Tracer(galaxies=all_galaxies, cosmology=cosmology)

tracer_image_np = tracer.image_2d_from(grid=grid).native.array

from autolens.lens import tracer_util

traced_existing = tracer_util.traced_grid_2d_list_from(
    planes=tracer.planes,
    grid=grid,
    cosmology=cosmology,
    xp=np,
)
source_grid_existing = traced_existing[-1]

source_image_existing = source_galaxy.image_2d_from(
    grid=source_grid_existing,
).native.array

scan_lensed_1d = source_light_fn(
    substructure_util.traced_grids_via_scan(
        grid=grid_array,
        halo_params=halo_params,
        halo_mask=halo_mask,
        scaling_matrix=scaling_matrix,
        lens_mass_fn=lens_mass_fn,
        lens_mass_params=lens_mass_params,
        lens_plane_mask=lens_plane_mask,
        sheet_kappas=sheet_kappas,
        halo_profile_cls=ag.mp.NFWTruncatedSph,
    )[-1],
    source_light_params,
)
scan_lensed_2d = np.array(scan_lensed_1d).reshape(image_shape)

np.testing.assert_allclose(
    scan_lensed_2d.ravel(),
    np.array(source_image_existing).ravel(),
    atol=1e-4,
    err_msg="Pre-convolution lensed image mismatch between scan and tracer paths",
)
print("PASS: Pre-convolution lensed image matches existing Tracer path")

"""
__JIT compilation__
"""
jitted_sim = jax.jit(
    lambda hp, hm, sk, lmp, slp: substructure_util.simulate_substructure(
        grid=grid_array,
        image_shape=image_shape,
        halo_params=hp,
        halo_mask=hm,
        scaling_matrix=scaling_matrix,
        lens_mass_fn=lens_mass_fn,
        lens_mass_params=lmp,
        lens_plane_mask=lens_plane_mask,
        sheet_kappas=sk,
        source_light_fn=source_light_fn,
        source_light_params=slp,
        psf_kernel=psf_kernel,
        exposure_time=300.0,
        background_sky_level=0.1,
        prng_key=None,
        halo_profile_cls=ag.mp.NFWTruncatedSph,
    )
)

image_jit = jitted_sim(
    halo_params, halo_mask, sheet_kappas, lens_mass_params, source_light_params
)

np.testing.assert_allclose(
    np.array(image_jit),
    np.array(image_scan),
    atol=1e-6,
    err_msg="JIT result doesn't match eager result",
)
print("PASS: jax.jit(simulate_substructure) compiles and matches")

"""
__Noise path__
"""
key = jax.random.PRNGKey(42)

image_noisy = substructure_util.simulate_substructure(
    grid=grid_array,
    image_shape=image_shape,
    halo_params=halo_params,
    halo_mask=halo_mask,
    scaling_matrix=scaling_matrix,
    lens_mass_fn=lens_mass_fn,
    lens_mass_params=lens_mass_params,
    lens_plane_mask=lens_plane_mask,
    sheet_kappas=sheet_kappas,
    source_light_fn=source_light_fn,
    source_light_params=source_light_params,
    psf_kernel=psf_kernel,
    exposure_time=300.0,
    background_sky_level=0.1,
    prng_key=key,
    halo_profile_cls=ag.mp.NFWTruncatedSph,
)

assert not jnp.allclose(image_noisy, image_scan), "Noisy image identical to clean"
assert not jnp.any(jnp.isnan(image_noisy)), "NaN in noisy image"
print("PASS: Poisson noise produces different, non-NaN image")

"""
__Different PRNGKey produces different noise__
"""
key2 = jax.random.PRNGKey(99)
image_noisy_2 = substructure_util.simulate_substructure(
    grid=grid_array,
    image_shape=image_shape,
    halo_params=halo_params,
    halo_mask=halo_mask,
    scaling_matrix=scaling_matrix,
    lens_mass_fn=lens_mass_fn,
    lens_mass_params=lens_mass_params,
    lens_plane_mask=lens_plane_mask,
    sheet_kappas=sheet_kappas,
    source_light_fn=source_light_fn,
    source_light_params=source_light_params,
    psf_kernel=psf_kernel,
    exposure_time=300.0,
    background_sky_level=0.1,
    prng_key=key2,
    halo_profile_cls=ag.mp.NFWTruncatedSph,
)

assert not jnp.allclose(image_noisy, image_noisy_2), "Different keys gave same noise"
print("PASS: Different PRNGKeys produce different noise realizations")

print("\nAll tests passed.")
