"""
Mesh Magnification: Areas And Magnification Against Simulator Truth
===================================================================

Every pixelized source magnification — the workspace `source_science.py` scripts and
PyAutoLens' `magnification` latent (`autolens/analysis/latent.py`) — is

    mu = sum(mapped_reconstructed_operated_data * pixel_area) / sum(s_i * A_i)

where `s_i` is the source reconstruction and `A_i` is
`mapper.mesh_geometry.areas_for_magnification`. If the areas are not the quadrature
weights of the mesh's reconstructed surface, every magnification is biased. This
happened twice: the Delaunay areas were Voronoi cells that biased mu by -13% to -99%
(PyAutoArray#522/#524), and `RectangularUniform` borrowed the adaptive meshes'
kernel-CDF guard-ring geometry, which does not describe its mapper at all, biasing
mu by -77% to +259% (PyAutoArray#659).

This script checks every mesh end to end against simulator truth (PyAutoArray#659):

 - **Lensed leg**: an elliptical Sersic source behind an `Isothermal` + shear lens. The
   true magnification is the lensed image-plane flux over the intrinsic source flux,
   both integrated on high-resolution grids from the simulator's own tracer.
 - **Identity leg**: the same source with no lens mass, so the true magnification is
   exactly 1.

For every mesh the dataset is fitted with the TRUE tracer and a fixed-regularization
pixelization (no search), and the script asserts:

 - the areas are finite, non-negative and sum to the area of the domain the mesh
   covers (the over-sampled data's bounding box for the rectangular meshes, the
   convex hull of the vertices for the Delaunay family);
 - the recovered mu agrees with the truth within the per-mesh tolerance below, on
   both legs;
 - the workspace formula and the PyAutoLens `magnification` latent agree.

A per-mesh table is printed on every run.

__Tolerances__

Measured with this script's configuration (PyAutoArray#659) and set per leg at ~1.5x
the measurement, floored at 1%. They are tight enough that the old `RectangularUniform` bias
(+259% / +27% on these legs before the fix), the old adaptive rectangular
node-midpoint bias (up to +14.43%), the old kNN dual-area bias (up to +6.10%) and
the old Delaunay Voronoi bias all fail.

`DelaunayNN` inherits the Delaunay dual areas, which approximate rather than equal
the quadrature of its Sibson interpolant; it measures within the 1% floor here.

__Env__

`full_datasets` is load-bearing: `PYAUTO_SMALL_DATASETS=1` caps the grids, masks and
image meshes this script builds to 15x15 at 0.6"/pixel, which cannot resolve the
source and would make every magnification a resolution artefact.

ENV: full_datasets
"""

# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import os
import time

import numpy as np
import scipy.spatial

import autolens as al
from autolens.analysis.latent import magnification as latent_magnification

assert os.environ.get("PYAUTO_SMALL_DATASETS") != "1", (
    "PYAUTO_SMALL_DATASETS=1 caps this script's grids to 15x15; the `full_datasets` "
    "ENV token must unset it."
)

_START = time.perf_counter()

"""
__Tolerances__

Absolute tolerance on `mu_recovered / mu_true - 1`, per mesh and per leg, as
`(lensed, identity)`. The measured errors are quoted beside each; the tolerances are
~1.5x the measurement (floored at 1%, the level at which noise and regularization
move the tight meshes: `Delaunay` itself lands at -0.25% / +0.43%).
"""
MU_TOLERANCE = {
    # Delaunay: exact quadrature of its linear interpolant. RectangularUniform: its
    # cells tile the overlay box exactly (sum-exact) but per node only approximate
    # the bilinear quadrature near the boundary ring, negligible for a source away
    # from the mesh edge. The adaptive rectangular meshes: exact per-axis quadrature
    # of each node's bilinear hat through the mapper's CDF (PyAutoArray#659; their
    # node-midpoint cells were off by up to +14.43% lensed / +11.23% identity).
    "RectangularUniform": (0.01, 0.01),  # measured -0.16%, -0.48%
    "Delaunay": (0.01, 0.01),  # measured -0.25%, +0.43%
    # unchanged by the exact areas (-1.82% before and after), so this is the
    # coarse density-adapted reconstruction under regularization, not the areas
    "RectangularRTUAdaptDensity": (0.03, 0.01),  # measured -1.82%, -0.26%
    "RectangularBilinearAdaptDensity": (0.01, 0.01),  # measured -0.61%, -0.34%
    "RectangularBilinearAdaptImage": (0.01, 0.01),  # measured -0.57%, -0.49%
    "RectangularRTUAdaptImage": (0.01, 0.01),  # measured +0.08%, +0.12%
    # Delaunay dual areas under the Sibson interpolant: an approximation, but
    # within the 1% floor here
    "DelaunayNN": (0.01, 0.01),  # measured -0.27%, -0.41%
    # quadrature of the kNN interpolants' own basis functions (PyAutoArray#665;
    # the Delaunay dual areas they used before measured +6.10% / -5.11% and
    # -1.71% / -3.57%)
    "KNearestNeighbor": (0.01, 0.01),  # measured -0.22%, -0.12%
    "KNNBarycentric": (0.01, 0.01),  # measured -0.37%, -0.14%
}
LEG_INDEX = {"lensed": 0, "identity": 1}

RECTANGULAR_SHAPE = (30, 30)
IMAGE_MESH_SHAPE = (26, 26)
EDGE_POINTS = 30

"""
__Dataset__

A 0.1"/pixel image with a 3.0" circular mask. The noise is low so that the recovered
magnification measures the areas, not the noise.
"""
pixel_scales = 0.1
mask_radius = 3.0

grid = al.Grid2D.uniform(
    shape_native=(80, 80), pixel_scales=pixel_scales, respect_small_datasets=False
)

psf = al.Convolver.from_gaussian(
    shape_native=(11, 11), sigma=0.1, pixel_scales=pixel_scales, normalize=True
)

simulator = al.SimulatorImaging(
    exposure_time=1.0e5,
    psf=psf,
    background_sky_level=0.1,
    add_poisson_noise_to_data=True,
    noise_seed=1,
)

source_galaxy_light = al.Galaxy(
    redshift=1.0,
    bulge=al.lp.Sersic(
        centre=(0.05, 0.03),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=60.0),
        intensity=1.0,
        effective_radius=0.25,
        sersic_index=1.0,
    ),
)

lens_galaxy = al.Galaxy(
    redshift=0.5,
    mass=al.mp.Isothermal(
        centre=(0.0, 0.0),
        einstein_radius=1.2,
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
    ),
)
field = al.MassField(
    redshift=0.5, shear=al.mp.ExternalShear(gamma_1=0.03, gamma_2=-0.02)
)

# The identity leg: a lens galaxy with no mass, so light rays are not deflected.
no_mass_galaxy = al.Galaxy(redshift=0.5)

LEGS = {
    "lensed": dict(galaxies=[lens_galaxy], fields=[field]),
    "identity": dict(galaxies=[no_mass_galaxy], fields=[]),
}


def tracer_from(leg, source_galaxy):
    return al.Tracer(
        galaxies=LEGS[leg]["galaxies"] + [source_galaxy], fields=LEGS[leg]["fields"]
    )


"""
__True Magnification__

The lensed flux is the source's image-plane surface brightness integrated on a
0.02"/pixel grid (over-sampled 4x4) covering the whole image; the intrinsic flux is
the source's surface brightness integrated on an equally fine source-plane grid. Both
come from the simulator's own tracer, so they are independent of any mesh.

The recovered magnification only sees flux inside the mask, so the script also checks
that the mask contains essentially all the lensed flux.
"""
fine_grid = al.Grid2D.uniform(
    shape_native=(300, 300),
    pixel_scales=0.02,
    over_sample_size=4,
    respect_small_datasets=False,
)
fine_mask = al.Mask2D.circular(
    shape_native=fine_grid.shape_native,
    pixel_scales=fine_grid.pixel_scales,
    radius=mask_radius - 0.2,
)
fine_area = fine_grid.pixel_area

intrinsic_flux = (
    np.sum(source_galaxy_light.image_2d_from(grid=fine_grid).array) * fine_area
)

truth = {}
for leg in LEGS:
    lensed_image = tracer_from(leg, source_galaxy_light).image_2d_from(grid=fine_grid)
    lensed_flux = np.sum(lensed_image.array) * fine_area
    in_mask = np.sum(lensed_image.native.array[~fine_mask]) * fine_area
    assert in_mask / lensed_flux > 0.999, (leg, in_mask / lensed_flux)
    truth[leg] = lensed_flux / intrinsic_flux

assert abs(truth["identity"] - 1.0) < 1.0e-6, truth["identity"]
print(f"[time] truth: {time.perf_counter() - _START:.1f}s")
print(
    f"true magnification: lensed {truth['lensed']:.4f}, identity {truth['identity']:.6f}"
)

"""
__Simulate And Mask__
"""
mask = al.Mask2D.circular(
    shape_native=grid.shape_native, pixel_scales=pixel_scales, radius=mask_radius
)

datasets = {}
adapt_images = {}
for leg in LEGS:
    tracer = tracer_from(leg, source_galaxy_light)
    dataset = simulator.via_tracer_from(tracer=tracer, grid=grid)
    dataset = dataset.apply_mask(mask=mask)
    dataset = dataset.apply_over_sampling(over_sample_size_pixelization=2)
    datasets[leg] = dataset

    # The noise-free lensed source is the adapt image the `*AdaptImage` meshes and
    # the Delaunay image mesh adapt to.
    adapt_images[leg] = tracer.image_2d_from(grid=dataset.grids.lp)


"""
__Meshes__

Every mesh in `al.mesh`. The rectangular meshes are 30x30. The Delaunay family uses
a 26x26 `Overlay` image-plane mesh grid traced to the source plane, plus a ring of 30
zeroed edge vertices just outside the mask, as the workspace examples do.
"""


def mesh_from(name, pixels):
    if name.startswith("Rectangular"):
        return getattr(al.mesh, name)(shape=RECTANGULAR_SHAPE)
    return getattr(al.mesh, name)(pixels=pixels, zeroed_pixels=EDGE_POINTS)


def fit_from(leg, name):
    dataset = datasets[leg]

    image_plane_mesh_grid = al.image_mesh.Overlay(
        shape=IMAGE_MESH_SHAPE, respect_small_datasets=False
    ).image_plane_mesh_grid_from(mask=dataset.mask)
    image_plane_mesh_grid = al.image_mesh.append_with_circle_edge_points(
        image_plane_mesh_grid=image_plane_mesh_grid,
        centre=mask.mask_centre,
        radius=mask_radius + mask.pixel_scale / 2.0,
        n_points=EDGE_POINTS,
    )

    pixelization = al.Pixelization(
        mesh=mesh_from(name, pixels=image_plane_mesh_grid.shape[0] - EDGE_POINTS),
        regularization=al.reg.Constant(coefficient=1.0),
    )
    source_galaxy = al.Galaxy(redshift=1.0, pixelization=pixelization)

    fit = al.FitImaging(
        dataset=dataset,
        tracer=tracer_from(leg, source_galaxy),
        adapt_images=al.AdaptImages(
            galaxy_image_dict={source_galaxy: adapt_images[leg]},
            galaxy_image_plane_mesh_grid_dict={source_galaxy: image_plane_mesh_grid},
        ),
    )
    return fit


def covered_area_from(name, mapper):
    """
    The analytic area of the domain the mesh's areas tile: the convex hull of the
    vertices (Delaunay family) or the over-sampled data's bounding box (the
    rectangular meshes: `RectangularUniform`'s overlay grid spans it, and the
    adaptive meshes integrate their basis functions over it).
    """
    if not name.startswith("Rectangular"):
        return scipy.spatial.ConvexHull(
            np.asarray(mapper.mesh_geometry.mesh_grid_xy)
        ).volume
    return np.prod(
        np.ptp(np.asarray(mapper.source_plane_data_grid.over_sampled.array), axis=0)
    )


"""
__Fits__

The magnification is computed exactly as `source_science.py` computes it, and again
through PyAutoLens' `magnification` latent, which must agree.
"""
rows = []
failures = []

for name in MU_TOLERANCE:
    row = {"mesh": name}
    for leg in LEGS:
        _t0 = time.perf_counter()
        fit = fit_from(leg, name)
        inversion = fit.inversion
        mapper = inversion.cls_list_from(cls=al.Mapper)[0]

        areas = np.asarray(mapper.mesh_geometry.areas_for_magnification)
        reconstruction = np.asarray(inversion.reconstruction_dict[mapper])
        mapped = inversion.mapped_reconstructed_operated_data

        assert np.all(np.isfinite(areas)), (name, leg)
        assert np.all(areas >= 0.0), (name, leg)

        area_error = areas.sum() / covered_area_from(name, mapper) - 1.0
        # 1e-6 absorbs the 1e-8 overlay buffer of `RectangularUniform`
        if abs(area_error) > 1.0e-6:
            failures.append(f"{name} {leg}: area sum off by {area_error:+.2e}")

        mu = np.sum(mapped.array * mapped.pixel_area) / np.sum(reconstruction * areas)
        # The latent integrates the un-convolved mapped source image
        # (`galaxy_image_dict`), the workspace the PSF-convolved one, so they differ
        # by the PSF flux scattered beyond the mask edge (~1e-4 here).
        mu_latent = float(latent_magnification(fit=fit, magzero=25.0))
        if abs(mu_latent / mu - 1.0) > 1.0e-3:
            failures.append(f"{name} {leg}: latent mu {mu_latent} != workspace mu {mu}")

        mu_error = mu / truth[leg] - 1.0
        tolerance = MU_TOLERANCE[name][LEG_INDEX[leg]]
        if abs(mu_error) > tolerance:
            failures.append(
                f"{name} {leg}: mu {mu:.4f} vs truth {truth[leg]:.4f} "
                f"({mu_error:+.2%}, tolerance {tolerance:.1%})"
            )

        print(f"[time] {name} {leg}: {time.perf_counter() - _t0:.1f}s", flush=True)
        row[f"{leg}_mu"] = mu
        row[f"{leg}_err"] = mu_error
        row[f"{leg}_area_err"] = area_error
    rows.append(row)

"""
__Table__
"""
print()
print(
    f"{'mesh':33s} {'mu lensed':>10s} {'err':>8s} {'tol':>6s} {'mu identity':>12s} "
    f"{'err':>8s} {'tol':>6s} {'area err':>9s}"
)
for row in rows:
    print(
        f"{row['mesh']:33s} {row['lensed_mu']:10.4f} {row['lensed_err']:+8.2%} "
        f"{MU_TOLERANCE[row['mesh']][0]:6.1%} {row['identity_mu']:12.4f} "
        f"{row['identity_err']:+8.2%} {MU_TOLERANCE[row['mesh']][1]:6.1%} "
        f"{max(abs(row['lensed_area_err']), abs(row['identity_area_err'])):9.1e}"
    )
print(f"true mu: lensed {truth['lensed']:.4f}, identity {truth['identity']:.4f}")
print(f"[time] {time.perf_counter() - _START:.1f}s")

assert not failures, "\n".join(failures)
