"""
Cluster CSV API: Workspace Test
================================

Builds the truth cluster model entirely in Python and writes it out as a small set of CSVs that
downstream test scripts (``simulator.py``, ``likelihood_sanity.py``, etc.) consume. The pedagogical
guide for the CSV schema lives in ``autolens_workspace/scripts/cluster/csv_api.py`` — this script
is the test-workspace analogue, focused on producing a reproducible truth model.

The truth model covers every tier the cluster code path supports:

 - 2 main lens galaxies (dPIE mass + Sersic light, individually modelled)
 - 1 extra galaxy (dPIE mass + Sersic light, individually modelled — middle tier)
 - 1 host dark matter halo (NFWMCRLudlowSph, standalone)
 - 10 scaling-tier members (legacy 3-column scaling_galaxies.csv)
 - 2 background sources at different redshifts (SersicCore light + Point)

Outputs are written to ``dataset/cluster/test/``. The next script in the chain — ``simulator.py``
— loads these CSVs and runs the lensing simulation to produce ``data.fits`` + ``point_datasets.csv``.
"""

from autolens import jax_wrapper  # Sets JAX environment before other imports

from pathlib import Path

import autolens as al

"""
__Output Path__
"""
dataset_path = Path("dataset") / "cluster" / "test"
dataset_path.mkdir(exist_ok=True, parents=True)


"""
__Truth Model__

Concrete redshifts and centres for every galaxy in the truth model.
"""
redshift_lens = 0.5
source_redshifts = [1.0, 2.0]


"""
__Mass Profiles__

dPIE for the individually-modelled tier (mains + extra); NFW for the host halo. The scaling-tier
members carry their own simpler 3-column schema, written separately below.
"""
mass_profiles = {
    "lens_0": {
        "mass": al.mp.dPIEMassSph(
            centre=(0.0, 0.0),
            sigma=330.0,
            r_core=8.0,
            r_cut=20.0,
            redshift_object=0.5,
            redshift_source=2.0,
        ),
    },
    "lens_1": {
        "mass": al.mp.dPIEMassSph(
            centre=(10.0, 8.0),
            sigma=210.0,
            r_core=5.0,
            r_cut=12.0,
            redshift_object=0.5,
            redshift_source=2.0,
        ),
    },
    "extra_0": {
        "mass": al.mp.dPIEMassSph(
            centre=(-7.0, -4.0),
            sigma=135.0,
            r_core=2.0,
            r_cut=8.0,
            redshift_object=0.5,
            redshift_source=2.0,
        ),
    },
    "host_halo": {
        "dark": al.mp.NFWMCRLudlowSph(
            centre=(0.0, 0.0),
            mass_at_200=10**15.3,
            redshift_object=redshift_lens,
            redshift_source=max(source_redshifts),
        ),
    },
}


"""
__Light Profiles__

Sersic bulges for the lens tiers (used to simulate the imaging data + visualise the cluster).
Sources use the cored ``SersicCore`` so lensed arcs do not require explicit source-plane over-sampling.
"""
light_profiles = {
    "lens_0": {
        "bulge": al.lp.SersicSph(
            centre=(0.0, 0.0), intensity=1.5, effective_radius=3.0, sersic_index=4.0
        ),
    },
    "lens_1": {
        "bulge": al.lp.SersicSph(
            centre=(10.0, 8.0), intensity=0.8, effective_radius=1.5, sersic_index=3.5
        ),
    },
    "extra_0": {
        "bulge": al.lp.SersicSph(
            centre=(-7.0, -4.0), intensity=0.5, effective_radius=1.0, sersic_index=3.0
        ),
    },
    "source_0": {
        "bulge": al.lp.SersicCore(
            centre=(0.3, 0.5),
            ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=60.0),
            intensity=2.0,
            effective_radius=0.3,
            sersic_index=1.0,
        ),
    },
    "source_1": {
        "bulge": al.lp.SersicCore(
            centre=(-0.8, 1.2),
            ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=90.0),
            intensity=2.0,
            effective_radius=0.3,
            sersic_index=1.0,
        ),
    },
}


"""
__Point Models__

Each source carries a per-source ``Point`` model whose centre is paired to its multi-image positions
via name pairing (``point_i`` attribute on ``source_i`` Galaxy ↔ ``point_i`` row in point_datasets.csv).
"""
point_profiles = {
    "source_0": {"point_0": al.ps.Point(centre=(0.3, 0.5))},
    "source_1": {"point_1": al.ps.Point(centre=(-0.8, 1.2))},
}


"""
__Redshifts per Galaxy__
"""
redshifts_by_galaxy = {
    "lens_0": redshift_lens,
    "lens_1": redshift_lens,
    "extra_0": redshift_lens,
    "host_halo": redshift_lens,
    "source_0": source_redshifts[0],
    "source_1": source_redshifts[1],
}


"""
__Write Family CSVs__

One CSV per profile family (mass / light / point), each row carrying ``galaxy, attr_name,
profile_class, <params...>, redshift``.
"""
al.galaxy_models_to_csv(
    profiles_by_galaxy=mass_profiles,
    file_path=dataset_path / "mass.csv",
    family="mass",
    redshifts=redshifts_by_galaxy,
)

al.galaxy_models_to_csv(
    profiles_by_galaxy=light_profiles,
    file_path=dataset_path / "light.csv",
    family="light",
    redshifts=redshifts_by_galaxy,
)

al.galaxy_models_to_csv(
    profiles_by_galaxy=point_profiles,
    file_path=dataset_path / "point.csv",
    family="point",
    redshifts=redshifts_by_galaxy,
)


"""
__Scaling Galaxies__

10 lower-mass members on the legacy 3-column schema. Truth scaling relation parameters used by
``simulator.py`` are ``scaling_factor = 0.3`` and ``scaling_exponent = 1.0`` (so per-member b0
equals luminosity).
"""
scaling_galaxies_centres = [
    (5.5, -6.5),
    (-7.5, 3.0),
    (12.0, -5.0),
    (-4.0, -9.0),
    (3.0, 13.0),
    (-14.0, 4.0),
    (15.0, 9.0),
    (-9.0, -12.0),
    (8.5, 5.5),
    (-6.5, 11.0),
]

scaling_galaxies_luminosities = [
    0.40,
    0.32,
    0.25,
    0.20,
    0.16,
    0.13,
    0.10,
    0.08,
    0.06,
    0.05,
]

al.galaxy_table_to_csv(
    centres=scaling_galaxies_centres,
    luminosities=scaling_galaxies_luminosities,
    file_path=dataset_path / "scaling_galaxies.csv",
)


"""
__Summary__
"""
print(f"Wrote cluster truth model CSVs to {dataset_path}:")
print(f"  mass.csv             — {len(mass_profiles)} galaxies")
print(f"  light.csv            — {len(light_profiles)} galaxies")
print(f"  point.csv            — {len(point_profiles)} sources")
print(f"  scaling_galaxies.csv — {len(scaling_galaxies_luminosities)} members")
print(
    f"Next: scripts/cluster/simulator.py reads these and produces the lensing dataset."
)
