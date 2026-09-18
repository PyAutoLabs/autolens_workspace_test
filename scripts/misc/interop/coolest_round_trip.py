"""
Interop: COOLEST template round trip
====================================

This script validates the COOLEST interop layer (``al.interop.coolest``,
PyAutoLens#613) end-to-end:

1. **Export / import round trip (legacy galaxy-attached shear)** — a PowerLaw
   + ExternalShear lens with Sersic light and a Sersic source (the standard
   cross-code parity model) is written to a COOLEST ``.json`` template via
   ``to_coolest`` and read back via ``from_coolest``; tracer deflections and
   images must be numerically identical.

   The shear here is attached to the lens ``Galaxy``, which is the *legacy*
   form — this case is deliberately kept as the regression for the exporter's
   legacy peel (it must still split the galaxy-attached shear out into its own
   ``MassField`` entity). ``misc/mass/galaxy_attached_legacy.py`` and this case
   are the only two galaxy-attached sites allowed in either workspace.

2. **Export / import round trip (``MassField``)** — the same system with the
   shear held in an ``al.MassField`` in the tracer's ``fields=`` slot, which is
   the current API. It must export to the same 2 ``Galaxy`` + 1 ``MassField``
   entities and import back as an ``al.MassField`` in ``tracer_back.fields``.

3. **Convention checks** — the written template's ``theta_E`` carries the
   COOLEST intermediate-axis factor ``sqrt(q) (2/(1+q))^(1/(gamma-1))``, the
   position angle is East-of-North, and shear is stored as a ``MassField``
   entity.

4. **NFW round trip** — the physical ``rho_c`` normalization converts back to
   the input ``kappa_s`` exactly when the same cosmology is used on both
   sides.

Note that ``from_coolest`` returns the shear as an ``al.MassField`` in
``tracer_back.fields`` in *both* cases: a COOLEST ``MassField`` entity has no
galaxy to belong to, so the import is one-way — galaxy-attached in, field out.

Requires the optional ``coolest`` package (``pip install autolens[coolest]``).
"""

import json
import os
import tempfile

import numpy as np
import numpy.testing as npt

import autolens as al

"""
__Round trip: PowerLaw + Shear lens, Sersic light, Sersic source (LEGACY galaxy-attached shear)__

The shear is attached to the lens `Galaxy` here on purpose: this is the regression for the exporter's legacy
peel, which must lift a galaxy-attached external field out into its own COOLEST `MassField` entity. Do not
migrate it to `fields=` — the `MassField` form is covered by its own case below.
"""
lens = al.Galaxy(
    redshift=0.5,
    bulge=al.lp.Sersic(
        centre=(0.05, -0.03),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=70.0),
        intensity=1.2,
        effective_radius=0.9,
        sersic_index=3.5,
    ),
    mass=al.mp.PowerLaw(
        centre=(0.05, -0.03),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.7, angle=45.0),
        einstein_radius=1.3,
        slope=2.1,
    ),
    shear=al.mp.ExternalShear(gamma_1=0.02, gamma_2=-0.03),
)
source = al.Galaxy(
    redshift=1.5,
    bulge=al.lp.Sersic(
        centre=(0.1, 0.2),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.6, angle=-30.0),
        intensity=0.7,
        effective_radius=0.3,
        sersic_index=1.2,
    ),
)

tracer = al.Tracer(galaxies=[lens, source])

grid = al.Grid2D.uniform(shape_native=(50, 50), pixel_scales=0.08)

with tempfile.TemporaryDirectory() as tmp_dir:
    file_path = al.interop.coolest.to_coolest(
        galaxies=tracer, file_path=os.path.join(tmp_dir, "template")
    )

    tracer_back = al.interop.coolest.from_coolest(file_path=file_path)

    npt.assert_allclose(
        tracer_back.deflections_yx_2d_from(grid=grid).array,
        tracer.deflections_yx_2d_from(grid=grid).array,
        rtol=1e-6,
        atol=1e-10,
    )
    npt.assert_allclose(
        tracer_back.image_2d_from(grid=grid).array,
        tracer.image_2d_from(grid=grid).array,
        rtol=1e-6,
        atol=1e-12,
    )
    print("PASS: tracer deflections + image round trip numerically identical")

    """
    The import is one-way: a COOLEST `MassField` entity has no galaxy to belong to, so the shear that went in
    attached to the lens `Galaxy` comes back as an `al.MassField` in `tracer_back.fields`. Follow it there —
    `tracer_back.galaxies` no longer has a `shear` attribute at all.
    """
    assert len(tracer_back.fields) == 1
    assert isinstance(tracer_back.fields[0], al.MassField)
    assert tracer_back.fields[0].redshift == 0.5

    # COOLEST carries no component *names*, so the imported profiles are named `mass_{i}` positionally.
    shear_back = tracer_back.fields[0].mass_0

    assert isinstance(shear_back, al.mp.ExternalShear)

    npt.assert_allclose(shear_back.gamma_1, 0.02, rtol=0, atol=1e-12)
    npt.assert_allclose(shear_back.gamma_2, -0.03, rtol=0, atol=1e-12)

    lens_back = [g for g in tracer_back.galaxies if g.redshift == 0.5][0]

    assert not hasattr(lens_back, "shear")
    print(
        "PASS: galaxy-attached shear imports back as an al.MassField in tracer.fields"
    )

    """
    __Convention checks on the written template__
    """
    with open(file_path) as f:
        template = json.load(f)

    entities = template["lensing_entities"]
    types = [entity["type"] for entity in entities]
    assert types.count("Galaxy") == 2, types
    assert types.count("MassField") == 1, types
    print("PASS: shear exported as a MassField entity")

    lens_entity = [
        e for e in entities if e["type"] == "Galaxy" and e["redshift"] == 0.5
    ][0]
    pemd_parameters = lens_entity["mass_model"][0]["parameters"]

    theta_e_expected = 1.3 * np.sqrt(0.7) * (2.0 / 1.7) ** (1.0 / 1.1)
    npt.assert_allclose(
        pemd_parameters["theta_E"]["point_estimate"]["value"],
        theta_e_expected,
        rtol=1e-10,
    )
    npt.assert_allclose(
        pemd_parameters["phi"]["point_estimate"]["value"], -45.0, rtol=1e-10
    )
    print("PASS: theta_E intermediate-axis factor + East-of-North angle")

"""
__Round trip: the same system with the shear in a `MassField` (current API)__

The lens galaxy keeps only its own light and mass; the external shear goes into an `al.MassField` at the lens
redshift, handed to the tracer's `fields=` argument. The exported template must have the same entity make-up as
the legacy case above — 2 `Galaxy` + 1 `MassField` — and the import must give the shear back in
`tracer_back.fields`.
"""
lens_no_shear = al.Galaxy(
    redshift=0.5,
    bulge=lens.bulge,
    mass=lens.mass,
)

field = al.MassField(
    redshift=0.5, shear=al.mp.ExternalShear(gamma_1=0.02, gamma_2=-0.03)
)

tracer_field = al.Tracer(galaxies=[lens_no_shear, source], fields=[field])

with tempfile.TemporaryDirectory() as tmp_dir:
    file_path = al.interop.coolest.to_coolest(
        galaxies=tracer_field, file_path=os.path.join(tmp_dir, "template_field")
    )

    with open(file_path) as f:
        template_field = json.load(f)

    types_field = [entity["type"] for entity in template_field["lensing_entities"]]

    assert types_field.count("Galaxy") == 2, types_field
    assert types_field.count("MassField") == 1, types_field

    tracer_field_back = al.interop.coolest.from_coolest(file_path=file_path)

    assert len(tracer_field_back.fields) == 1
    assert isinstance(tracer_field_back.fields[0], al.MassField)
    assert tracer_field_back.fields[0].redshift == 0.5

    shear_field_back = tracer_field_back.fields[0].mass_0

    assert isinstance(shear_field_back, al.mp.ExternalShear)

    npt.assert_allclose(shear_field_back.gamma_1, 0.02, rtol=0, atol=1e-12)
    npt.assert_allclose(shear_field_back.gamma_2, -0.03, rtol=0, atol=1e-12)

    npt.assert_allclose(
        tracer_field_back.deflections_yx_2d_from(grid=grid).array,
        tracer_field.deflections_yx_2d_from(grid=grid).array,
        rtol=1e-6,
        atol=1e-10,
    )
    print("PASS: MassField round trip — 1:1 entity mapping, shear + deflection parity")

"""
The two constructions are numerically the same system, which is why the legacy case above may keep its
galaxy-attached shear without weakening anything: the tracer sums every deflection field over the plane.
"""
npt.assert_allclose(
    tracer_field.deflections_yx_2d_from(grid=grid).array,
    tracer.deflections_yx_2d_from(grid=grid).array,
    rtol=0,
    atol=1e-12,
)
print("PASS: galaxy-attached and MassField tracers deflect identically")

"""
__NFW round trip (same cosmology both directions)__
"""
cosmology = al.cosmo.Planck15()

galaxies = [
    al.Galaxy(
        redshift=0.3,
        mass=al.mp.NFW(
            centre=(0.0, 0.1),
            ell_comps=al.convert.ell_comps_from(axis_ratio=0.85, angle=10.0),
            kappa_s=0.15,
            scale_radius=6.0,
        ),
    ),
    al.Galaxy(redshift=1.0, bulge=al.lp.SersicSph(intensity=0.5)),
]

with tempfile.TemporaryDirectory() as tmp_dir:
    file_path = al.interop.coolest.to_coolest(
        galaxies=galaxies,
        file_path=os.path.join(tmp_dir, "template_nfw"),
        cosmology=cosmology,
    )
    tracer_back = al.interop.coolest.from_coolest(
        file_path=file_path, cosmology=cosmology
    )

nfw_back = [g for g in tracer_back.galaxies if g.redshift == 0.3][0].mass_0

npt.assert_allclose(nfw_back.kappa_s, 0.15, rtol=1e-8)
npt.assert_allclose(nfw_back.scale_radius, 6.0, rtol=1e-8)
print("PASS: NFW kappa_s / scale_radius round trip via sigma_crit")

print("All COOLEST round-trip checks passed.")
