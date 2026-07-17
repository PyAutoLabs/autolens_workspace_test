"""
Interop: herculens power-law parity via COOLEST
===============================================

Cross-code validation of the "direct link" between the PyAutoLens
``PowerLawIntermediate`` mass profile and herculens's ``EPL`` power-law
(PyAutoLens#616): both parameterise the Einstein radius in the COOLEST
intermediate-axis convention, so the **same COOLEST template parameters**
must produce numerically matching lensing quantities in both codes with an
identity parameter mapping (no sqrt(q) factors).

The script:

1. Exports a ``PowerLawIntermediate`` lens model to a COOLEST ``.json``
   template via ``al.interop.coolest.to_coolest`` and asserts the written
   ``theta_E`` equals the profile's ``einstein_radius`` exactly (identity
   mapping).

2. Builds a herculens ``EPL`` directly from the template's PEMD parameters
   (converting only the COOLEST East-of-North position angle to
   herculens/lenstronomy's x-axis ellipticity components).

3. Asserts convergence and deflection angles from the two codes match on a
   shared grid of coordinates (herculens works in (x, y); PyAutoLens in
   (y, x)).

4. Repeats the convergence check for the spherical (q = 1) limit and for the
   round trip ``from_coolest(intermediate=True)``.

Requires the optional ``coolest`` package and ``herculens`` (installed in
this environment only — herculens is never a PyAutoLens dependency).
"""

import json
import os
import tempfile

import numpy as np
import numpy.testing as npt

import autolens as al

from herculens.MassModel.mass_model import MassModel


def herculens_kwargs_from(pemd_parameters):
    """
    herculens/lenstronomy EPL kwargs from COOLEST PEMD parameters: theta_E,
    gamma and the centre map across unchanged; the East-of-North position
    angle converts to an x-axis angle (+90 deg) before forming the
    lenstronomy-convention ellipticity components.
    """
    q = pemd_parameters["q"]
    phi_x = np.radians(pemd_parameters["phi"] + 90.0)
    f = (1.0 - q) / (1.0 + q)
    return dict(
        theta_E=pemd_parameters["theta_E"],
        gamma=pemd_parameters["gamma"],
        e1=f * np.cos(2.0 * phi_x),
        e2=f * np.sin(2.0 * phi_x),
        center_x=pemd_parameters["center_x"],
        center_y=pemd_parameters["center_y"],
    )


def point_estimates_from(template, entity_index=0, profile_index=0):
    profile = template["lensing_entities"][entity_index]["mass_model"][profile_index]
    return {
        name: parameter["point_estimate"]["value"]
        for name, parameter in profile["parameters"].items()
    }


# Shared evaluation coordinates: herculens (x, y) vs PyAutoLens (y, x).
x = np.array([0.5, 0.7, -1.1, 0.9, -0.4, 1.3])
y = np.array([1.0, -0.3, 0.2, 0.8, -1.2, 0.1])
grid_al = al.Grid2DIrregular(np.stack([y, x], axis=-1))

"""
__Elliptical power-law__
"""
mass = al.mp.PowerLawIntermediate(
    centre=(0.05, -0.03),
    ell_comps=al.convert.ell_comps_from(axis_ratio=0.7, angle=45.0),
    einstein_radius=1.2,
    slope=2.1,
)
lens = al.Galaxy(redshift=0.5, mass=mass)
source = al.Galaxy(redshift=1.5, bulge=al.lp.SersicSph(intensity=0.5))

with tempfile.TemporaryDirectory() as tmp_dir:
    file_path = al.interop.coolest.to_coolest(
        galaxies=[lens, source], file_path=os.path.join(tmp_dir, "template")
    )
    with open(file_path) as f:
        template = json.load(f)

    tracer_back = al.interop.coolest.from_coolest(
        file_path=file_path, intermediate=True
    )

pemd = point_estimates_from(template)

npt.assert_allclose(pemd["theta_E"], 1.2, rtol=1e-12)
npt.assert_allclose(pemd["gamma"], 2.1, rtol=1e-12)
print("PASS: COOLEST theta_E == PowerLawIntermediate.einstein_radius (identity)")

mass_back = [g for g in tracer_back.galaxies if g.redshift == 0.5][0].mass_0
assert type(mass_back).__name__ == "PowerLawIntermediate"
npt.assert_allclose(mass_back.einstein_radius, 1.2, rtol=1e-12)
print("PASS: from_coolest(intermediate=True) round trip theta_E identity")

mass_model_herculens = MassModel(["EPL"])
kwargs = [herculens_kwargs_from(pemd)]

kappa_herculens = np.asarray(mass_model_herculens.kappa(x, y, kwargs))
kappa_al = np.asarray(mass.convergence_2d_from(grid=grid_al))

npt.assert_allclose(kappa_al, kappa_herculens, rtol=1e-8)
print("PASS: convergence parity (herculens EPL vs PowerLawIntermediate)")

alpha_x_herculens, alpha_y_herculens = mass_model_herculens.alpha(x, y, kwargs)
deflections_al = np.asarray(mass.deflections_yx_2d_from(grid=grid_al))

npt.assert_allclose(deflections_al[:, 1], np.asarray(alpha_x_herculens), rtol=1e-6)
npt.assert_allclose(deflections_al[:, 0], np.asarray(alpha_y_herculens), rtol=1e-6)
print("PASS: deflection-angle parity (herculens EPL vs PowerLawIntermediate)")

"""
__Spherical (q = 1) limit__
"""
mass_sph = al.mp.PowerLawIntermediate(
    centre=(0.0, 0.0), ell_comps=(0.0, 0.0), einstein_radius=1.0, slope=2.0
)
kwargs_sph = [
    dict(theta_E=1.0, gamma=2.0, e1=0.0, e2=0.0, center_x=0.0, center_y=0.0)
]

kappa_herculens_sph = np.asarray(mass_model_herculens.kappa(x, y, kwargs_sph))
kappa_al_sph = np.asarray(mass_sph.convergence_2d_from(grid=grid_al))

npt.assert_allclose(kappa_al_sph, kappa_herculens_sph, rtol=1e-8)
print("PASS: spherical-limit convergence parity")

print("All herculens <-> PyAutoLens COOLEST power-law parity checks passed.")
