"""
Interop: COOLEST template round trip
====================================

This script validates the COOLEST interop layer (``al.interop.coolest``,
PyAutoLens#613) end-to-end:

1. **Export / import round trip** — a PowerLaw + ExternalShear lens with
   Sersic light and a Sersic source (the standard cross-code parity model) is
   written to a COOLEST ``.json`` template via ``to_coolest`` and read back
   via ``from_coolest``; tracer deflections and images must be numerically
   identical.

2. **Convention checks** — the written template's ``theta_E`` carries the
   COOLEST intermediate-axis factor ``sqrt(q) (2/(1+q))^(1/(gamma-1))``, the
   position angle is East-of-North, and shear is stored as a ``MassField``
   entity.

3. **NFW round trip** — the physical ``rho_c`` normalization converts back to
   the input ``kappa_s`` exactly when the same cosmology is used on both
   sides.

Requires the optional ``coolest`` package (``pip install autolens[coolest]``).
"""

import json
import os
import tempfile

import numpy as np
import numpy.testing as npt

import autolens as al

"""
__Round trip: PowerLaw + Shear lens, Sersic light, Sersic source__
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
