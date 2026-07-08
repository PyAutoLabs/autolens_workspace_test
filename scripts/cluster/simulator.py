"""
Cluster Simulator: Workspace Test
==================================

Loads the truth model CSVs produced by ``csv_api.py`` and runs the lensing simulation, emitting:

 - ``data.fits`` / ``noise_map.fits`` / ``psf.fits`` — CCD imaging of the cluster
 - ``point_datasets.csv`` — multi-image positions per source, with redshifts
 - ``tracer.json`` — the truth ``Tracer`` (for downstream visualization / sanity checks)

Smaller than ``autolens_workspace/scripts/cluster/simulator.py`` — workspace_test conventions:
deterministic noise seed, smaller imaging grid (250×250 vs 1000×1000) for speed, and PointSolver
JIT for the point-position solve.

The truth model is read entirely from CSVs; no parameter values are duplicated in this script.
Re-running it after editing the CSVs produces a new dataset reflecting whatever you edited.
"""

from autoconf import jax_wrapper  # Sets JAX environment before other imports

import jax
import jax.numpy as jnp
import numpy as np
from pathlib import Path

import autofit as af
import autolens as al
import autolens.plot as aplt
from autolens.jax import register_tracer_classes

from autoarray.abstract_ndarray import register_instance_pytree
from autolens.lens.tracer import Tracer


"""
__Dataset Paths__
"""
dataset_path = Path("dataset") / "cluster" / "test"
dataset_path.mkdir(exist_ok=True, parents=True)


"""
__Truth Model from CSVs__

Load the family tables written by ``csv_api.py``. ``galaxies_from_csv_tables`` joins the rows by
galaxy name and builds concrete ``Galaxy`` instances with each profile attached under its
``attr_name``. The scaling tier uses its legacy 3-column CSV.
"""
mass_table = al.galaxy_models_from_csv(dataset_path / "mass.csv", family="mass")
light_table = al.galaxy_models_from_csv(dataset_path / "light.csv", family="light")
point_table = al.galaxy_models_from_csv(dataset_path / "point.csv", family="point")

galaxies_by_name = al.galaxies_from_csv_tables(mass_table, light_table, point_table)

main_lens_galaxies = [
    galaxies_by_name["lens_0"],
    galaxies_by_name["lens_1"],
    galaxies_by_name["extra_0"],
]
host_halo_galaxy = galaxies_by_name["host_halo"]
source_galaxies = [galaxies_by_name["source_0"], galaxies_by_name["source_1"]]


"""
__Scaling Galaxies__

The 10 lower-mass members each get a ``dPIEMassSph`` derived from the scaling relation
``b0 = scaling_factor * luminosity ** scaling_exponent``. Core/truncation radii are held fixed
across the tier — matches the truth values used by ``autolens_workspace/scripts/cluster/simulator.py``.
"""
scaling_galaxies_table = al.galaxy_table_from_csv(dataset_path / "scaling_galaxies.csv")
scaling_galaxies_centres = list(scaling_galaxies_table.centres.in_list)
scaling_galaxies_luminosities = scaling_galaxies_table.luminosities

scaling_factor_truth = 0.3
scaling_exponent_truth = 1.0
scaling_ra = 0.1
scaling_rs = 10.0

scaling_galaxies = []
for centre, luminosity in zip(scaling_galaxies_centres, scaling_galaxies_luminosities):
    b0 = scaling_factor_truth * luminosity**scaling_exponent_truth
    scaling_galaxies.append(
        al.Galaxy(
            redshift=0.5,
            mass=al.mp.dPIEMassSph(
                centre=tuple(centre), ra=scaling_ra, rs=scaling_rs, b0=b0
            ),
        )
    )


"""
__Tracer__
"""
tracer = al.Tracer(
    galaxies=main_lens_galaxies
    + scaling_galaxies
    + [host_halo_galaxy]
    + source_galaxies
)


"""
__JAX JIT__

Register every concrete class reachable from the tracer as a JAX pytree node so ``PointSolver.solve``
can be wrapped in ``jax.jit``. Same pattern as ``autolens_workspace/scripts/cluster/simulator.py``.
"""
redshift_lens = 0.5
source_redshifts = [g.redshift for g in source_galaxies]

_lens_models = [
    af.Model(
        al.Galaxy,
        redshift=redshift_lens,
        bulge=af.Model(
            al.lp.SersicSph,
            centre=g.bulge.centre,
            intensity=g.bulge.intensity,
            effective_radius=g.bulge.effective_radius,
            sersic_index=g.bulge.sersic_index,
        ),
        mass=af.Model(
            al.mp.dPIEMassSph,
            centre=g.mass.centre,
            ra=g.mass.ra,
            rs=g.mass.rs,
            b0=g.mass.b0,
        ),
    )
    for g in main_lens_galaxies
]

_halo_model = af.Model(
    al.Galaxy,
    redshift=redshift_lens,
    dark=af.Model(
        al.mp.NFWMCRLudlowSph,
        centre=host_halo_galaxy.dark.centre,
        mass_at_200=host_halo_galaxy.dark.mass_at_200,
        redshift_object=redshift_lens,
        redshift_source=max(source_redshifts),
    ),
)

_source_models = [
    af.Model(
        al.Galaxy,
        redshift=g.redshift,
        bulge=af.Model(
            al.lp.SersicCore,
            centre=g.bulge.centre,
            ell_comps=g.bulge.ell_comps,
            intensity=g.bulge.intensity,
            effective_radius=g.bulge.effective_radius,
            sersic_index=g.bulge.sersic_index,
        ),
        **{f"point_{i}": af.Model(al.ps.Point, centre=getattr(g, f"point_{i}").centre)},
    )
    for i, g in enumerate(source_galaxies)
]

_registration_model = af.Collection(
    galaxies=af.Collection(*(_lens_models + [_halo_model] + _source_models))
)

register_instance_pytree(Tracer, no_flatten=("cosmology",))
register_tracer_classes(tracer)


"""
__Point Solver__

Smaller starting grid than the workspace simulator (500×500 @ 0.1") for speed under
``PYAUTO_TEST_MODE``. ``magnification_threshold=0.1`` discards the central image.
"""
solver = al.PointSolver.for_grid(
    grid=al.Grid2D.uniform(shape_native=(500, 500), pixel_scales=0.1),
    pixel_scale_precision=0.001,
    magnification_threshold=0.1,
)


jitted_solve = jax.jit(
    lambda source_plane_coordinate: solver.solve(
        tracer=tracer,
        source_plane_coordinate=source_plane_coordinate,
        xp=jnp,
        remove_infinities=False,
    ).array
)


positions_list = []
for i, src_galaxy in enumerate(source_galaxies):
    src_centre = src_galaxy.bulge.centre
    coord = jnp.asarray(src_centre)
    raw = np.asarray(jitted_solve(coord))
    finite = ~(np.isinf(raw).any(axis=1) | np.isnan(raw).any(axis=1))
    positions_list.append(al.Grid2DIrregular(raw[finite]))


"""
__Point Datasets__
"""
position_noise = 0.005

dataset_list = []
for i, positions in enumerate(positions_list):
    dataset_list.append(
        al.PointDataset(
            name=f"point_{i}",
            positions=positions,
            positions_noise_map=position_noise,
            redshift=source_redshifts[i],
        )
    )

al.output_to_csv(
    datasets=dataset_list,
    file_path=dataset_path / "point_datasets.csv",
)

for i, dataset in enumerate(dataset_list):
    al.output_to_json(
        obj=dataset,
        file_path=dataset_path / f"point_dataset_{i}.json",
    )


"""
__Tracer JSON__
"""
al.output_to_json(obj=tracer, file_path=dataset_path / "tracer.json")


"""
__Imaging__

CCD imaging used to visualise the cluster + drive the imaging-mode likelihood tests in
``likelihood_imaging.py``. 500×500 @ 0.1" — smaller than the workspace simulator's 1000×1000 to keep
the test workspace fast.
"""
imaging_grid = al.Grid2D.uniform(shape_native=(500, 500), pixel_scales=0.1)

imaging_over_sample = al.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=imaging_grid,
    sub_size_list=[16, 4, 2],
    radial_list=[0.3, 0.6],
    centre_list=[tuple(g.mass.centre) for g in main_lens_galaxies]
    + scaling_galaxies_centres,
)

imaging_grid = imaging_grid.apply_over_sampling(over_sample_size=imaging_over_sample)

psf = al.Convolver.from_gaussian(
    shape_native=(11, 11), sigma=0.1, pixel_scales=imaging_grid.pixel_scales
)

simulator = al.SimulatorImaging(
    exposure_time=300.0,
    psf=psf,
    background_sky_level=0.1,
    add_poisson_noise_to_data=True,
    noise_seed=1,
)

dataset = simulator.via_tracer_from(tracer=tracer, grid=imaging_grid)

aplt.fits_imaging(
    dataset=dataset,
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    overwrite=True,
)


"""
__Summary__
"""
print(f"Cluster test dataset written to {dataset_path}:")
print(f"  data.fits / noise_map.fits / psf.fits — imaging")
print(
    f"  point_datasets.csv — {sum(len(p.positions) for p in dataset_list)} multi-images across {len(dataset_list)} sources"
)
print(f"  tracer.json — truth Tracer")
