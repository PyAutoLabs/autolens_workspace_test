"""
Parity: DatasetModel shift+rotation vs profile-baked shift+rotation (lp_linear)
==============================================================================

Sanity check that ``DatasetModel.grid_offset`` + ``DatasetModel.grid_rotation_angle``
produce a fit that is mathematically equivalent to baking the same shift and
rotation into the light/mass profile parameters directly.

The setup:

  * Simulate two noiseless imaging datasets of the SAME physical lens system,
    observed with different telescope alignments.

      - ``dataset_0``: lens at origin, mass at angle 0 deg, source at origin.
      - ``dataset_1``: every profile's centre is shifted by ``offset`` and every
        elliptical profile's position angle is rotated by ``theta``. This emulates
        a different roll-angle pointing of the same physical scene.

  * For each dataset, run two fit variants and assert the log-likelihood agrees:

      A. ``profile_baked``: fit with profile parameters set to the truth in the
         dataset's own frame. ``DatasetModel`` is default (identity).
      B. ``dataset_model``: fit with profiles in the reference (``dataset_0``)
         frame and a ``DatasetModel(grid_offset=offset, grid_rotation_angle=theta)``
         that pulls the data-grid back into the reference frame before model
         evaluation.

  * Cross-dataset assertion: ``A0 == A1`` and ``B0 == B1`` (same physical scene
    + noiseless ⇒ same log-likelihood regardless of telescope alignment).

  * Within-dataset assertion: ``A1 == B1`` (the ``DatasetModel`` route reproduces
    the profile-baked route exactly).

If any of those four equalities fails, the rotation feature has a bug.

Uses ``lp_linear.Sersic`` for both lens and source so the intensity is solved via
inversion (the realistic JWST modelling path), exercising the full FitImaging
inversion chain.
"""

import os
import numpy as np
import autoarray as aa
import autogalaxy as ag
import autolens as al


OFFSET = (0.3, 0.2)
THETA = 12.0  # degrees, CCW.


def _ell_comps_rotated_by(axis_ratio, angle_deg, delta_deg):
    """Helper: ell_comps at ``angle_deg + delta_deg``."""
    return al.convert.ell_comps_from(axis_ratio=axis_ratio, angle=angle_deg + delta_deg)


def _rotate_centre(centre, offset, angle_deg):
    """Apply shift then rotate (matches Grid2D.subtracted_and_rotated_from) to a centre.

    NB: this is the OPPOSITE of what we want for "bake the truth into the profile in
    the dataset's frame". When the data grid is rotated CCW by theta about offset
    relative to the reference frame, a point at (cy, cx) in the reference frame
    appears in the data frame at the position you get by ROTATING (cy, cx) by
    +theta about offset and then ADDING the offset back in. We invert
    ``subtracted_and_rotated_from`` here.
    """
    cy, cx = centre
    oy, ox = offset
    cos_t = np.cos(np.deg2rad(angle_deg))
    sin_t = np.sin(np.deg2rad(angle_deg))
    # Inverse of: shift by -offset then rotate by +angle CCW.
    # Forward rotation: (y', x') = (sx sin + sy cos, sx cos - sy sin) where (sy, sx) = (y - oy, x - ox)
    # Inverse: given (y', x'), recover (y, x):
    #   sy = y' cos - x' sin; sx = x' cos + y' sin; y = sy + oy; x = sx + ox.
    # But the simulator runs the FORWARD direction: data centre = inverse_transform(reference centre).
    # We pre-rotate the reference centre by +theta about offset then add offset, giving the
    # apparent centre as observed in the rotated data frame.
    y_rot = cx * sin_t + cy * cos_t + oy
    x_rot = cx * cos_t - cy * sin_t + ox
    return (y_rot, x_rot)


def make_dataset(galaxies, grid):
    """Simulate a noiseless imaging dataset with unit noise map."""
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


# ------------------------------------------------------------------
# Reference frame (dataset_0) profile parameters — the "truth".
# ------------------------------------------------------------------
LENS_CENTRE_REF = (0.0, 0.0)
LENS_ELL_ANGLE_REF = 0.0
SOURCE_CENTRE_REF = (0.05, 0.05)
SOURCE_ELL_ANGLE_REF = 30.0

axis_ratio_lens = 0.7
axis_ratio_source = 0.8
einstein_radius = 1.2

# Data-frame parameters (dataset_1) — same physical scene viewed through a
# rotated+shifted coordinate frame.
LENS_CENTRE_DATA = _rotate_centre(LENS_CENTRE_REF, OFFSET, THETA)
SOURCE_CENTRE_DATA = _rotate_centre(SOURCE_CENTRE_REF, OFFSET, THETA)
LENS_ELL_ANGLE_DATA = LENS_ELL_ANGLE_REF + THETA
SOURCE_ELL_ANGLE_DATA = SOURCE_ELL_ANGLE_REF + THETA


def build_lens_source(lens_centre, lens_angle, source_centre, source_angle):
    """Build a (lens, source) tuple in a given reference frame."""
    lens = al.Galaxy(
        redshift=0.5,
        bulge=al.lp_linear.Sersic(
            centre=lens_centre,
            ell_comps=al.convert.ell_comps_from(
                axis_ratio=axis_ratio_lens, angle=lens_angle
            ),
            effective_radius=0.4,
            sersic_index=3.0,
        ),
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
        bulge=al.lp_linear.SersicCore(
            centre=source_centre,
            ell_comps=al.convert.ell_comps_from(
                axis_ratio=axis_ratio_source, angle=source_angle
            ),
            effective_radius=0.1,
            sersic_index=1.0,
        ),
    )
    return lens, source


grid = al.Grid2D.uniform(shape_native=(51, 51), pixel_scales=0.1, over_sample_size=1)

# Build the two datasets. ``dataset_0`` uses reference-frame profile centres;
# ``dataset_1`` uses data-frame centres (shifted+rotated copies of the reference).
lens_ref, src_ref = build_lens_source(
    LENS_CENTRE_REF, LENS_ELL_ANGLE_REF, SOURCE_CENTRE_REF, SOURCE_ELL_ANGLE_REF
)
tracer_sim_0 = al.Tracer(galaxies=[lens_ref, src_ref])
dataset_0 = make_dataset([lens_ref, src_ref], grid)

lens_data, src_data = build_lens_source(
    LENS_CENTRE_DATA, LENS_ELL_ANGLE_DATA, SOURCE_CENTRE_DATA, SOURCE_ELL_ANGLE_DATA
)
tracer_sim_1 = al.Tracer(galaxies=[lens_data, src_data])
dataset_1 = make_dataset([lens_data, src_data], grid)

mask = al.Mask2D.circular(
    shape_native=dataset_0.data.shape_native, pixel_scales=0.1, radius=2.0
)
masked_0 = dataset_0.apply_mask(mask=mask)
masked_1 = dataset_1.apply_mask(mask=mask)


# ------------------------------------------------------------------
# Variant A: profile-baked fits (no DatasetModel)
# ------------------------------------------------------------------
fit_A0 = al.FitImaging(dataset=masked_0, tracer=tracer_sim_0)
fit_A1 = al.FitImaging(dataset=masked_1, tracer=tracer_sim_1)

# ------------------------------------------------------------------
# Variant B: identity-frame profiles + DatasetModel applies the shift+rotation
# ------------------------------------------------------------------
fit_B0 = al.FitImaging(
    dataset=masked_0,
    tracer=tracer_sim_0,
    dataset_model=al.DatasetModel(),
)
fit_B1 = al.FitImaging(
    dataset=masked_1,
    tracer=tracer_sim_0,  # same reference-frame tracer as A0
    dataset_model=al.DatasetModel(grid_offset=OFFSET, grid_rotation_angle=THETA),
)


# ------------------------------------------------------------------
# Report and assert
# ------------------------------------------------------------------
print("=== lp_linear DatasetModel parity ===")
print(
    f"  A0 (dataset_0, profile-baked) : log_likelihood = {fit_A0.log_likelihood:.10e}, chi^2 = {fit_A0.chi_squared:.3e}"
)
print(
    f"  A1 (dataset_1, profile-baked) : log_likelihood = {fit_A1.log_likelihood:.10e}, chi^2 = {fit_A1.chi_squared:.3e}"
)
print(
    f"  B0 (dataset_0, DatasetModel)  : log_likelihood = {fit_B0.log_likelihood:.10e}, chi^2 = {fit_B0.chi_squared:.3e}"
)
print(
    f"  B1 (dataset_1, DatasetModel)  : log_likelihood = {fit_B1.log_likelihood:.10e}, chi^2 = {fit_B1.chi_squared:.3e}"
)

# Cross-dataset parity: same physical scene + noiseless ⇒ same log-likelihood.
np.testing.assert_allclose(
    fit_A0.log_likelihood,
    fit_A1.log_likelihood,
    atol=1.0e-6,
    err_msg="A0 != A1: profile-baked fits to two datasets of the same scene disagree.",
)
np.testing.assert_allclose(
    fit_B0.log_likelihood,
    fit_B1.log_likelihood,
    atol=1.0e-6,
    err_msg="B0 != B1: DatasetModel fits to two datasets of the same scene disagree.",
)

# Within-dataset parity: DatasetModel reproduces profile-baked exactly.
np.testing.assert_allclose(
    fit_A0.log_likelihood,
    fit_B0.log_likelihood,
    atol=1.0e-6,
    err_msg="A0 != B0: DatasetModel default identity differs from no-DatasetModel.",
)
np.testing.assert_allclose(
    fit_A1.log_likelihood,
    fit_B1.log_likelihood,
    atol=1.0e-6,
    err_msg="A1 != B1: DatasetModel rotation+shift fit differs from profile-baked fit.",
)

print(
    "All four log-likelihoods agree to 1e-6 — lp_linear DatasetModel parity confirmed."
)
