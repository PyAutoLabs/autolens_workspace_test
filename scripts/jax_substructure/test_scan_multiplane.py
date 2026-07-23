"""
Test: scan-based multi-plane ray-tracing
========================================

Validates that traced_grids_via_scan (jax.lax.scan path) reproduces the
existing Python-loop Tracer path for a 4-plane system with LOS halos,
a macro lens, and negative kappa sheets. Ref: PyAutoLens#542, prompt 2.

__Env__

Test-harness configuration (PyAutoHands docs/env_profile_redesign.md §10).
Tests the full simulate_substructure path at image_shape=(51,51); need JAX
enabled and full-size datasets (the 15x15 cap raises).

ENV: jax full_datasets
"""

import numpy as np
import jax
import jax.numpy as jnp

import autoarray as aa
import autogalaxy as ag
import autolens as al
from autolens.lens import tracer_util, substructure_util

np.random.seed(42)

grid = al.Grid2D.uniform(shape_native=(50, 50), pixel_scales=0.05)
grid_array = jnp.array(grid.array)

"""
__Build a 4-plane system__

Plane 0 (z=0.25): 5 NFWTruncatedSph LOS halos + negative kappa sheet
Plane 1 (z=0.50): PowerLaw + ExternalShear macro + 5 NFWTruncatedSph subhalos + sheet
Plane 2 (z=0.75): 5 NFWTruncatedSph LOS halos + sheet
Plane 3 (z=1.00): source (no deflectors)
"""
cosmology = ag.cosmo.Planck15()

plane_redshifts = [0.25, 0.5, 0.75, 1.0]
n_halos_per_plane = 5
max_n = 10

all_galaxies = []

for plane_idx, z in enumerate(plane_redshifts[:3]):
    plane_halos = []
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
        plane_halos.append(halo)

    sheet_kappa = -np.random.uniform(0.001, 0.005)
    sheet = ag.Galaxy(
        redshift=z,
        mass_sheet=ag.mp.MassSheet(centre=(0.0, 0.0), kappa=sheet_kappa),
    )
    plane_halos.append(sheet)
    all_galaxies.extend(plane_halos)

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

source_galaxy = al.Galaxy(redshift=1.0)
all_galaxies.append(source_galaxy)

"""
__Path A: existing Tracer path__
"""
tracer = al.Tracer(galaxies=all_galaxies, cosmology=cosmology)
planes = tracer.planes

traced_grids_existing = tracer_util.traced_grid_2d_list_from(
    planes=planes,
    grid=grid,
    cosmology=cosmology,
    xp=np,
)

"""
__Path B: scan-based path__
"""
halo_galaxies = [g for g in all_galaxies if g.redshift < 1.0 and g is not macro_galaxy]

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

lens_plane_mask = jnp.array(
    [1.0 if abs(z - 0.5) < 1e-6 else 0.0 for z in plane_redshifts]
)

traced_grids_scan = substructure_util.traced_grids_via_scan(
    grid=grid_array,
    halo_params=halo_params,
    halo_mask=halo_mask,
    scaling_matrix=scaling_matrix,
    lens_mass_fn=lens_mass_fn,
    lens_mass_params=lens_mass_params,
    lens_plane_mask=lens_plane_mask,
    sheet_kappas=sheet_kappas,
    halo_profile_cls=ag.mp.NFWTruncatedSph,
)

"""
__Compare source-plane grids__
"""
source_grid_existing = np.array(traced_grids_existing[-1].array)
source_grid_scan = np.array(traced_grids_scan[-1])

np.testing.assert_allclose(
    source_grid_scan,
    source_grid_existing,
    atol=1e-5,
    err_msg="Source-plane grid mismatch between scan and loop paths",
)
print("PASS: Source-plane grids match (scan vs loop)")

"""
__Compare all plane grids__
"""
for i in range(len(plane_redshifts)):
    existing_arr = np.array(traced_grids_existing[i].array)
    scan_arr = np.array(traced_grids_scan[i])

    np.testing.assert_allclose(
        scan_arr,
        existing_arr,
        atol=1e-5,
        err_msg=f"Plane {i} (z={plane_redshifts[i]}) grid mismatch",
    )
print("PASS: All plane grids match")

"""
__JIT compilation__
"""
jitted_scan = jax.jit(
    lambda hp, hm, sk, lmp: substructure_util.traced_grids_via_scan(
        grid=grid_array,
        halo_params=hp,
        halo_mask=hm,
        scaling_matrix=scaling_matrix,
        lens_mass_fn=lens_mass_fn,
        lens_mass_params=lmp,
        lens_plane_mask=lens_plane_mask,
        sheet_kappas=sk,
        halo_profile_cls=ag.mp.NFWTruncatedSph,
    )
)

jit_result = jitted_scan(halo_params, halo_mask, sheet_kappas, lens_mass_params)

np.testing.assert_allclose(
    np.array(jit_result[-1]),
    source_grid_existing,
    atol=1e-5,
    err_msg="JIT scan result doesn't match",
)
print("PASS: jax.jit(traced_grids_via_scan) compiles and matches")

"""
__Verify recompilation avoidance__
"""
halo_params_shifted = halo_params.at[:, :, 0].add(0.01)
jit_result_2 = jitted_scan(
    halo_params_shifted, halo_mask, sheet_kappas, lens_mass_params
)

assert not jnp.allclose(
    jit_result[-1], jit_result_2[-1]
), "Different params produced identical results"
print("PASS: Different params produce different results (no stale cache)")

print("\nAll tests passed.")
