"""
Parity: DatasetModel shift+rotation vs profile-baked shift+rotation (Delaunay)
==============================================================================

Companion to ``dataset_model_parity_lp_linear.py`` but with a Delaunay
pixelization source. The Delaunay case is a much stronger sanity check: the
log-likelihood depends not just on the data fit (chi^2) but also on the
source-plane mesh shape and the regularization terms — none of which are zero,
so the parity is non-trivial.

The setup:

  * ``dataset_0``: lens at origin, mass at angle 0 deg, source pattern centred
    near the reference frame origin. This IS the reference frame.
  * ``dataset_1``: every profile centre shifted by ``OFFSET`` and every
    elliptical angle rotated by ``THETA``. Same physical scene as dataset_0,
    just observed in a frame that has been shifted+rotated relative to the
    reference. This emulates a JWST roll-angle misalignment.

  * For each dataset, two fit variants:

      A. ``profile_baked``: every profile in the dataset's own native frame.
         The Delaunay image-plane mesh-grid lives in the dataset's own frame
         (built off the dataset's mask, then forward-transformed to the data
         frame for ``dataset_1``). No ``DatasetModel`` involved.
      B. ``dataset_model``: every profile in the REFERENCE frame. For
         ``dataset_1`` the cached image-plane mesh-grid starts in the DATA
         frame (mimicking what a previous adapt-image fit on ``dataset_1``
         would have cached); ``DatasetModel(grid_offset=OFFSET,
         grid_rotation_angle=-THETA)`` then pulls both the data grid AND the
         mesh-grid back to the reference frame — which is the exact code path
         ``AdaptImages.updated_via_instance_from`` exercises in production.

  * Asserts: ``A0 ≈ A1 ≈ B0 ≈ B1`` log-likelihoods agree to the discretisation
    floor (~0.2 absolute on log-likelihoods near -1144).

This is the test that catches the "weird source reconstruction" failure mode
Qiuhan saw on ``dev_Q``. Without rotating the cached mesh-grid in lockstep
with the data grid, B1 would diverge from A1 by ~1 or more in log-likelihood
even for the small (12 degree) rotation used here.
"""
import numpy as np
import autoarray as aa
import autolens as al


OFFSET = (0.3, 0.2)
THETA = 12.0   # degrees, CCW.

LENS_CENTRE_REF = (0.0, 0.0)
LENS_ELL_ANGLE_REF = 0.0
SOURCE_CENTRE_REF = (0.05, 0.05)
SOURCE_ELL_ANGLE_REF = 30.0

axis_ratio_lens = 0.7
axis_ratio_source = 0.8
einstein_radius = 1.2

OVERLAY_SHAPE = (20, 20)
EDGE_PIXELS = 30
REGULARIZATION_COEFFICIENT = 1.0


def _forward_transform_centre(centre, offset, angle_deg):
    """Map a reference-frame centre into the data frame via
    ``data = R(+angle) @ ref + offset`` (rotate then shift)."""
    cy, cx = centre
    cos_t = np.cos(np.deg2rad(angle_deg))
    sin_t = np.sin(np.deg2rad(angle_deg))
    y_data = cx * sin_t + cy * cos_t + offset[0]
    x_data = cx * cos_t - cy * sin_t + offset[1]
    return (y_data, x_data)


def _forward_transform_grid(grid_array, offset, angle_deg):
    """Map every ``(y, x)`` row of ``grid_array`` from ref frame to data frame:
    ``data = R(+angle) @ ref + offset``. Returns a raw ndarray."""
    cos_t = np.cos(np.deg2rad(angle_deg))
    sin_t = np.sin(np.deg2rad(angle_deg))
    my = grid_array[:, 0]
    mx = grid_array[:, 1]
    y_data = mx * sin_t + my * cos_t + offset[0]
    x_data = mx * cos_t - my * sin_t + offset[1]
    return np.stack((y_data, x_data), axis=-1)


def make_dataset(galaxies, grid):
    psf = aa.Convolver.from_gaussian(
        shape_native=(3, 3), sigma=0.05, pixel_scales=grid.pixel_scales[0]
    )
    simulator = al.SimulatorImaging(
        exposure_time=300.0, psf=psf, add_poisson_noise_to_data=False
    )
    dataset = simulator.via_galaxies_from(galaxies=galaxies, grid=grid)
    dataset.noise_map = aa.Array2D.ones(
        shape_native=dataset.data.shape_native, pixel_scales=grid.pixel_scales
    )
    return dataset


def build_sim_galaxies(lens_centre, lens_angle, source_centre, source_angle):
    """Galaxies for simulating the data: parametric source we will reconstruct
    with a Delaunay pixelization in the fit."""
    lens = al.Galaxy(
        redshift=0.5,
        mass=al.mp.Isothermal(
            centre=lens_centre,
            einstein_radius=einstein_radius,
            ell_comps=al.convert.ell_comps_from(
                axis_ratio=axis_ratio_lens, angle=lens_angle
            ),
        ),
    )
    source = al.Galaxy(
        redshift=1.0,
        bulge=al.lp.Sersic(
            centre=source_centre,
            ell_comps=al.convert.ell_comps_from(
                axis_ratio=axis_ratio_source, angle=source_angle
            ),
            intensity=0.3,
            effective_radius=0.15,
            sersic_index=1.0,
        ),
    )
    return [lens, source]


def build_fit_galaxies(lens_centre, lens_angle):
    """Lens at supplied (centre, angle), Delaunay-source for inversion."""
    lens = al.Galaxy(
        redshift=0.5,
        mass=al.mp.Isothermal(
            centre=lens_centre,
            einstein_radius=einstein_radius,
            ell_comps=al.convert.ell_comps_from(
                axis_ratio=axis_ratio_lens, angle=lens_angle
            ),
        ),
    )
    pixelization = al.Pixelization(
        mesh=al.mesh.Delaunay(
            pixels=OVERLAY_SHAPE[0] * OVERLAY_SHAPE[1] + EDGE_PIXELS,
            zeroed_pixels=EDGE_PIXELS,
        ),
        regularization=al.reg.ConstantSplit(coefficient=REGULARIZATION_COEFFICIENT),
    )
    source = al.Galaxy(redshift=1.0, pixelization=pixelization)
    return lens, source


def build_image_plane_mesh_grid(mask):
    """Build a Delaunay image-plane mesh-grid from an Overlay image-mesh over
    the given mask, plus a ring of edge pixels."""
    image_mesh = al.image_mesh.Overlay(shape=OVERLAY_SHAPE)
    mesh_grid = image_mesh.image_plane_mesh_grid_from(mask=mask)
    mesh_grid = al.image_mesh.append_with_circle_edge_points(
        image_plane_mesh_grid=mesh_grid,
        centre=mask.mask_centre,
        radius=2.0 + mask.pixel_scale / 2.0,
        n_points=EDGE_PIXELS,
    )
    return mesh_grid


grid = al.Grid2D.uniform(shape_native=(51, 51), pixel_scales=0.1, over_sample_size=1)

# Data-frame centres / angles.
LENS_CENTRE_DATA = _forward_transform_centre(LENS_CENTRE_REF, OFFSET, THETA)
SOURCE_CENTRE_DATA = _forward_transform_centre(SOURCE_CENTRE_REF, OFFSET, THETA)
LENS_ELL_ANGLE_DATA = LENS_ELL_ANGLE_REF + THETA
SOURCE_ELL_ANGLE_DATA = SOURCE_ELL_ANGLE_REF + THETA

dataset_0 = make_dataset(
    build_sim_galaxies(LENS_CENTRE_REF, LENS_ELL_ANGLE_REF, SOURCE_CENTRE_REF, SOURCE_ELL_ANGLE_REF),
    grid,
)
dataset_1 = make_dataset(
    build_sim_galaxies(LENS_CENTRE_DATA, LENS_ELL_ANGLE_DATA, SOURCE_CENTRE_DATA, SOURCE_ELL_ANGLE_DATA),
    grid,
)

mask = al.Mask2D.circular(
    shape_native=dataset_0.data.shape_native, pixel_scales=0.1, radius=2.0
)
masked_0 = dataset_0.apply_mask(mask=mask)
masked_1 = dataset_1.apply_mask(mask=mask)

settings = al.Settings(use_border_relocator=True)

# A reference image-plane mesh-grid in the REFERENCE frame, built once.
mesh_grid_ref = build_image_plane_mesh_grid(masked_0.mask)
# The same mesh forward-transformed into the DATA frame for dataset_1's native fit.
mesh_grid_data = aa.Grid2DIrregular(
    _forward_transform_grid(mesh_grid_ref.array, OFFSET, THETA)
)

# ------------------------------------------------------------------
# Variant A0: dataset_0, profile-baked at reference frame, no DatasetModel.
# ------------------------------------------------------------------
lens_A0, src_A0 = build_fit_galaxies(LENS_CENTRE_REF, LENS_ELL_ANGLE_REF)
tracer_A0 = al.Tracer(galaxies=[lens_A0, src_A0])
adapt_A0 = al.AdaptImages(
    galaxy_image_plane_mesh_grid_dict={src_A0: mesh_grid_ref}
)
fit_A0 = al.FitImaging(
    dataset=masked_0, tracer=tracer_A0, adapt_images=adapt_A0, settings=settings
)

# ------------------------------------------------------------------
# Variant A1: dataset_1, profile-baked at data frame, mesh in data frame, no
# DatasetModel. This is the "directly model the scene in the data frame"
# approach a user would take without DatasetModel.
# ------------------------------------------------------------------
lens_A1, src_A1 = build_fit_galaxies(LENS_CENTRE_DATA, LENS_ELL_ANGLE_DATA)
tracer_A1 = al.Tracer(galaxies=[lens_A1, src_A1])
adapt_A1 = al.AdaptImages(
    galaxy_image_plane_mesh_grid_dict={src_A1: mesh_grid_data}
)
fit_A1 = al.FitImaging(
    dataset=masked_1, tracer=tracer_A1, adapt_images=adapt_A1, settings=settings
)

# ------------------------------------------------------------------
# Variant B0: dataset_0, reference-frame tracer + DatasetModel default. Mesh
# starts in reference frame (would-be cached from a previous adapt fit on
# dataset_0).
# ------------------------------------------------------------------
lens_B0, src_B0 = build_fit_galaxies(LENS_CENTRE_REF, LENS_ELL_ANGLE_REF)
tracer_B0 = al.Tracer(galaxies=[lens_B0, src_B0])
adapt_B0 = al.AdaptImages(
    galaxy_image_plane_mesh_grid_dict={src_B0: mesh_grid_ref}
)
fit_B0 = al.FitImaging(
    dataset=masked_0,
    tracer=tracer_B0,
    adapt_images=adapt_B0,
    dataset_model=al.DatasetModel(),
    settings=settings,
)

# ------------------------------------------------------------------
# Variant B1: dataset_1, reference-frame tracer + DatasetModel(OFFSET, -THETA).
# Cached mesh starts in the DATA frame (mimicking having been cached from a
# previous adapt fit on dataset_1). We apply the same shift+rotate transform
# that AdaptImages.updated_via_instance_from would apply at Analysis time,
# pulling it back into the reference frame so it aligns with the ref-frame
# data grid produced by FitDataset.grids.
# ------------------------------------------------------------------
lens_B1, src_B1 = build_fit_galaxies(LENS_CENTRE_REF, LENS_ELL_ANGLE_REF)
tracer_B1 = al.Tracer(galaxies=[lens_B1, src_B1])
mesh_grid_B1 = mesh_grid_data.subtracted_and_rotated_from(
    offset=OFFSET, angle=-THETA
)
adapt_B1 = al.AdaptImages(
    galaxy_image_plane_mesh_grid_dict={src_B1: mesh_grid_B1}
)
fit_B1 = al.FitImaging(
    dataset=masked_1,
    tracer=tracer_B1,
    adapt_images=adapt_B1,
    dataset_model=al.DatasetModel(grid_offset=OFFSET, grid_rotation_angle=-THETA),
    settings=settings,
)

# ------------------------------------------------------------------
# Report and assert.
# ------------------------------------------------------------------
print("=== Delaunay DatasetModel parity ===")
print(f"  A0 (dataset_0, profile-baked) : log_likelihood = {fit_A0.log_likelihood:.6e}")
print(f"  A1 (dataset_1, profile-baked) : log_likelihood = {fit_A1.log_likelihood:.6e}")
print(f"  B0 (dataset_0, DatasetModel ) : log_likelihood = {fit_B0.log_likelihood:.6e}")
print(f"  B1 (dataset_1, DatasetModel ) : log_likelihood = {fit_B1.log_likelihood:.6e}")

# Delaunay parity has a floor set by image-plane pixel sampling: even with
# matched source-plane mesh topology, sampling the rotated brightness pattern
# at the rotated pixel grid introduces a small (~0.2 in log_likelihood)
# discretisation residual. Anything > ~0.5 indicates a real misalignment.
TOL_BETWEEN_FRAMES = 2.0e-1
TOL_DATASET_MODEL_VS_PROFILE = 5.0e-3   # B0 vs A0 and B1 vs A1 should match tightly.

np.testing.assert_allclose(
    fit_A0.log_likelihood, fit_A1.log_likelihood, atol=TOL_BETWEEN_FRAMES,
    err_msg="Delaunay A0 != A1: profile-baked fits to two datasets of the same physical scene disagree by more than the pixel-sampling floor.",
)
np.testing.assert_allclose(
    fit_B0.log_likelihood, fit_B1.log_likelihood, atol=TOL_BETWEEN_FRAMES,
    err_msg="Delaunay B0 != B1: DatasetModel fits to two datasets of the same physical scene disagree by more than the pixel-sampling floor.",
)
np.testing.assert_allclose(
    fit_A0.log_likelihood, fit_B0.log_likelihood, atol=TOL_DATASET_MODEL_VS_PROFILE,
    err_msg="Delaunay A0 != B0: DatasetModel default identity differs from no-DatasetModel.",
)
np.testing.assert_allclose(
    fit_A1.log_likelihood, fit_B1.log_likelihood, atol=TOL_DATASET_MODEL_VS_PROFILE,
    err_msg="Delaunay A1 != B1: DatasetModel rotation+shift fit differs from profile-baked fit (THIS IS THE BUG THE FIX TARGETS).",
)

print(f"All four log-likelihoods agree within tolerances — Delaunay DatasetModel parity confirmed.")
