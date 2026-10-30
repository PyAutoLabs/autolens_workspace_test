"""
Area Magnification: Arc Magnifications Against Forward-Traced Truth
===================================================================

The area magnification of an arc (LEGGOS II, arXiv:2606.20804, Eq. 6-7) is

    mu_area = A_img / A_src,    A_src = sum_i A_pix,i / |mu_i|

the area of an image-plane region over the area of the source-plane region it maps
from, with `mu_i` the local (Hessian) magnification of each pixel. `LensCalc` computes it
for a mask covering one image (`area_magnification_from`), splits a mask covering several
images into one mask per image (`image_masks_from`, by parity and connectivity) and
returns one value per image (`area_magnification_list_from`).

This script checks those against forward-traced truth:

 - The source is a uniform disk of known area, placed so it is well clear of every
   caustic. Each image therefore maps from the whole disk, and its true magnification
   is its image-plane area over the disk's area.
 - The true image areas come from ray-tracing a 0.004"/pixel grid back to the source
   plane and counting the points that land inside the disk, image by image.
 - The recovered values come from an image-plane mask of the pixels whose area is
   mostly inside the traced disk, at 0.1", 0.05" and 0.025"/pixel.

Two configurations: a "double" (the disk outside the tangential caustic, two images)
and a "quad" (the disk inside it, four images). At 0.05" and 0.025" the script asserts
the number of images, each image's magnification and the source-plane area each image
maps from (which should be the disk's area), within the tolerances below. At 0.1" it
only reports: small images there are a few pixels, and those of the quad break into
several pieces that the parity + connectivity split cannot reassemble, a documented
limitation of `image_masks_from`.

__Tolerances__

Relative error on each image's area magnification, then on its source-plane area,
measured with this configuration and set ~1.5-2x above the measurement:
"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

import numpy as np
from scipy import ndimage

import autolens as al

TOLERANCE = {
    # (pixel scale, configuration): (mu, source area)
    # double: bright image -0.9%, faint image (7 pixels) -5.5%; A_src -1.7%, -5.3%
    (0.05, "double"): (0.10, 0.10),
    # quad: -0.2%, -1.2%, +0.8%, -0.1%; A_src +3.0%, -4.7%, -5.2%, +12.0%
    (0.05, "quad"): (0.03, 0.20),
    # double: +0.3%, -0.8%; A_src -0.2%, +6.4%
    (0.025, "double"): (0.02, 0.10),
    # quad: within 0.3%; A_src within 2.1%
    (0.025, "quad"): (0.01, 0.04),
}

PIXEL_SCALES = (0.1, 0.05, 0.025)
SUB_SIZE = 8
FIELD = 4.0

"""
__Lens__

An elliptical isothermal lens with external shear. Its tangential caustic is a diamond
about 0.6" x 0.5" across, so a disk can sit wholly inside it (four images) or wholly
outside it (two images) with a margin of more than its own radius.
"""
lens_galaxy = al.Galaxy(
    redshift=0.5,
    mass=al.mp.Isothermal(
        centre=(0.0, 0.0),
        einstein_radius=1.2,
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.7, angle=45.0),
    ),
)
field = al.MassField(
    redshift=0.5, shear=al.mp.ExternalShear(gamma_1=0.03, gamma_2=-0.02)
)
tracer = al.Tracer(galaxies=[lens_galaxy, al.Galaxy(redshift=1.0)], fields=[field])

lens_calc = al.LensCalc.from_tracer(tracer=tracer)

SOURCES = {
    "double": dict(centre=(0.45, 0.3), radius=0.1, images=2),
    "quad": dict(centre=(0.02, 0.01), radius=0.06, images=4),
}

"""
__Caustic Margin__

Every image maps from the whole disk only if the disk crosses no caustic, so the script
first checks that the disk's distance to the caustics exceeds its radius.
"""
caustic_grid = al.Grid2D.uniform(
    shape_native=(300, 300), pixel_scales=0.02, respect_small_datasets=False
)
caustics = np.vstack(
    [
        np.asarray(caustic)
        for caustic in lens_calc.tangential_caustic_list_from(grid=caustic_grid)
        + lens_calc.radial_caustic_list_from(grid=caustic_grid)
    ]
)

for name, source in SOURCES.items():
    margin = np.min(
        np.hypot(
            caustics[:, 0] - source["centre"][0], caustics[:, 1] - source["centre"][1]
        )
    )
    assert margin > 1.5 * source["radius"], (
        f'{name}: the disk is {margin:.3f}" from a caustic, too close for its radius '
        f"{source['radius']}\""
    )


def inside_disk(grid, source):
    traced = np.asarray(tracer.traced_grid_2d_list_from(grid=grid)[-1])
    return (
        np.hypot(traced[:, 0] - source["centre"][0], traced[:, 1] - source["centre"][1])
        < source["radius"]
    )


"""
__Truth__

The true images: ray-trace a 0.004"/pixel grid, keep the points inside the disk, split
them into connected images and measure each image's area and centroid.
"""
TRUTH_PIXEL_SCALE = 0.004

truth_grid = al.Grid2D.uniform(
    shape_native=(int(FIELD / TRUTH_PIXEL_SCALE), int(FIELD / TRUTH_PIXEL_SCALE)),
    pixel_scales=TRUTH_PIXEL_SCALE,
    over_sample_size=1,
    respect_small_datasets=False,
)
truth_yx = np.asarray(truth_grid.native.array)

truth = {}

for name, source in SOURCES.items():
    disk_area = np.pi * source["radius"] ** 2

    inside = inside_disk(truth_grid, source).reshape(truth_grid.shape_native)
    labels, total = ndimage.label(inside)

    assert total == source["images"], f"{name}: forward tracing found {total} images"

    truth[name] = [
        dict(
            mu=np.sum(labels == label) * TRUTH_PIXEL_SCALE**2 / disk_area,
            centre=truth_yx[labels == label].mean(axis=0),
        )
        for label in range(1, total + 1)
    ]

"""
__Recovered__

For each pixel scale, build the image-plane mask of pixels with at least half their
area inside the traced disk, split it into images and compute each image's area
magnification and source-plane area. Recovered images are matched to the true ones by
centroid.
"""
rows = []

for pixel_scale in PIXEL_SCALES:
    pixels = int(FIELD / pixel_scale)

    pixel_centres = np.asarray(
        al.Grid2D.uniform(
            shape_native=(pixels, pixels),
            pixel_scales=pixel_scale,
            over_sample_size=1,
            respect_small_datasets=False,
        ).array
    )

    # A grid SUB_SIZE times finer, binned back to the pixels, gives each pixel's fraction
    # inside the disk (built directly so smoke-test over-sampling caps cannot change it).
    sub_pixels = al.Grid2D.uniform(
        shape_native=(pixels * SUB_SIZE, pixels * SUB_SIZE),
        pixel_scales=pixel_scale / SUB_SIZE,
        over_sample_size=1,
        respect_small_datasets=False,
    )

    for name, source in SOURCES.items():
        disk_area = np.pi * source["radius"] ** 2

        fraction = (
            inside_disk(sub_pixels, source)
            .reshape(pixels, SUB_SIZE, pixels, SUB_SIZE)
            .mean(axis=(1, 3))
            .ravel()
        )

        mask = al.Mask2D(
            mask=(fraction < 0.5).reshape(pixels, pixels), pixel_scales=pixel_scale
        )

        image_masks = lens_calc.image_masks_from(mask=mask)
        mu_list = lens_calc.area_magnification_list_from(mask=mask, sub_size=4)

        recovered = []

        for image_mask, mu in zip(image_masks, mu_list):
            unmasked = ~np.asarray(image_mask).ravel()
            recovered.append(
                dict(
                    mu=mu,
                    source_area=lens_calc.source_plane_area_from(
                        mask=image_mask, sub_size=4
                    )
                    / disk_area,
                    centre=pixel_centres[unmasked].mean(axis=0),
                    pixels=int(unmasked.sum()),
                )
            )

        rows.append(
            dict(
                pixel_scale=pixel_scale,
                name=name,
                images=len(recovered),
                recovered=recovered,
            )
        )

"""
__Report And Assertions__
"""
print(
    f"\n{'scale':>6s} {'config':>7s} {'images':>7s}   "
    f"{'true mu':>8s} {'mu':>8s} {'err':>7s} {'A_src/A':>8s} {'pixels':>7s}"
)

for row in rows:
    name = row["name"]
    expected = SOURCES[name]["images"]
    key = (row["pixel_scale"], name)

    for true_image in truth[name]:
        match = min(
            row["recovered"],
            key=lambda image: np.hypot(*(image["centre"] - true_image["centre"])),
        )
        error = match["mu"] / true_image["mu"] - 1.0

        print(
            f"{row['pixel_scale']:6.3f} {name:>7s} {row['images']:4d}/{expected:<2d}   "
            f"{true_image['mu']:8.3f} {match['mu']:8.3f} {error:+7.2%} "
            f"{match['source_area']:8.3f} {match['pixels']:7d}"
        )

        if key in TOLERANCE:
            mu_tolerance, area_tolerance = TOLERANCE[key]

            assert abs(error) < mu_tolerance, (
                f"{key}: image at {true_image['centre']} mu {match['mu']:.3f} vs "
                f"true {true_image['mu']:.3f} ({error:+.2%}, tolerance {mu_tolerance:.0%})"
            )
            assert abs(match["source_area"] - 1.0) < area_tolerance, (
                f"{key}: image at {true_image['centre']} maps from "
                f"{match['source_area']:.3f} of the disk area "
                f"(tolerance {area_tolerance:.0%})"
            )

    if key in TOLERANCE:
        assert (
            row["images"] == expected
        ), f"{key}: split into {row['images']} images, expected {expected}"
