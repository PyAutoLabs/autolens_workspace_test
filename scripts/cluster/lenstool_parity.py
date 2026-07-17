"""
Lenstool dPIE Parity Check
==========================

Validate the Lenstool-native dPIE parameterization (the default ``dPIEMass`` / ``dPIEMassSph``
constructors and the general-cosmology ``dPIEMassB0.from_lenstool`` / ``dPIEMassB0Sph.from_lenstool``
converters) and the internal consistency of the dPIE port, so the "PyAutoLens for Lenstool users"
workflow can trust the parameter mapping end to end.

Conventions verified against the Lenstool C source (git-cral.univ-lyon1.fr/lenstool/lenstool):

 - ``b0 = 6 * pia_c2 * sigma_LT^2`` with ``pia_c2 = 648000 / c^2`` (``set_potfile.c``,
   ``constant.h``); Lenstool applies D_LS/D_S separately at deflection time (``e_grad.c``),
   PyAutoGalaxy folds it into ``b0``.
 - ``.par`` ellipticity is emass = (a^2-b^2)/(a^2+b^2), converted internally to
   epot = (1-q)/(1+q) (``set_lens.c``) — exactly PyAuto's |ell_comps| — before ``ci05f``.
 - ``core_radius`` / ``cut_radius`` (arcsec) map one-to-one to ``ra`` / ``rs``.

Legs:

 1. Conversion parity — ``from_lenstool`` output vs hand-computed b0 / ell_comps, and the
    sqrt(3/2) fiducial-vs-central sigma cross-check via ``cosmology.velocity_dispersion_from``.
 2. Axisymmetric cross-path — spherical deflection alpha(R) vs (2/R) * integral of the
    independently-evaluated convergence (two separately ported code paths must agree).
 3. Analytic aperture mass — for ra -> 0 the enclosed convergence integral has the closed
    form pi * b0 * (R + rs - sqrt(rs^2 + R^2)) (Eliasdottir et al. 2007; Bergamini et al.
    2019 App. C in mass units); compare against numerical integration of the convergence.
 4. Elliptical deflection vs potential gradient — finite-difference grad(psi) vs the ci05f
    deflections (the potential and deflection ports are independent translations).
 5. Hessian consistency — ``analytical_hessian_2d_from`` vs finite differences of the
    deflection field.
 6. Lenstool reference values — the port's original hard-coded validation points
    (generated with Lenstool at porting time) re-asserted through the ``from_lenstool``
    construction path.

Regenerating true Lenstool references (optional, needs a Lenstool build):
    potentiel O1
        profil 81
        x_centre 0.5      # Lenstool x -> PyAuto x (tangent plane)
        y_centre -0.7     # Lenstool y -> PyAuto y
        ellipticite 0.0
        angle_pos 0.0
        core_radius 2.0
        cut_radius 3.0
        v_disp <sigma such that 6*648000*(sigma/c)^2 * dlsds = 5.2>
    then compare image-plane deflections from `runlt` against leg 6's values.
"""

import numpy as np

import autogalaxy as ag
import autolens as al

C_KM_S = 299792.458


def print_leg(name, max_frac):
    print(f"  {name}: max fractional deviation = {max_frac:.3e}")


"""
__Leg 1: Conversion parity__
"""
print("Leg 1: from_lenstool conversion parity")

cosmology = ag.cosmo.Planck15()

sigma_lt = 1200.0
z_l, z_s = 0.375, 2.0

mp_sph = al.mp.dPIEMassB0Sph.from_lenstool(
    sigma=sigma_lt,
    r_core=0.5,
    r_cut=40.0,
    redshift_object=z_l,
    redshift_source=z_s,
    cosmology=cosmology,
)

d_s = cosmology.angular_diameter_distance_to_earth_in_kpc_from(redshift=z_s)
d_ls = cosmology.angular_diameter_distance_between_redshifts_in_kpc_from(
    redshift_0=z_l, redshift_1=z_s
)

b0_expected = 6.0 * 648000.0 * (sigma_lt / C_KM_S) ** 2 * (d_ls / d_s)

assert abs(mp_sph.b0 / b0_expected - 1.0) < 1e-10
print(f"  b0({sigma_lt} km/s, z_l={z_l}, z_s={z_s}) = {mp_sph.b0:.4f} arcsec  OK")

# sqrt(3/2) fiducial-vs-central check through the independent isothermal relation.
sigma_0 = cosmology.velocity_dispersion_from(
    redshift_0=z_l, redshift_1=z_s, einstein_radius=mp_sph.b0
)
assert abs(sigma_0 / (sigma_lt * np.sqrt(1.5)) - 1.0) < 1e-6
print(f"  sigma_0 = {sigma_0:.2f} km/s = sqrt(3/2) * sigma_LT  OK")

# Ellipticity: emass -> epot equals Lenstool's set_lens.c conversion.
for emass in (0.05, 0.2, 0.4, 0.6):
    mp_ell = al.mp.dPIEMassB0.from_lenstool(ellipticity=emass, angle_pos=30.0)
    epot_lenstool = (1.0 - np.sqrt(1.0 - emass**2)) / emass
    assert abs(mp_ell._ellip() / epot_lenstool - 1.0) < 1e-10
print("  emass -> epot = (1-sqrt(1-e^2))/e = |ell_comps|  OK for e in {0.05..0.6}")

# Default-class equivalence: the default (Lenstool-native) constructor with flat H0/Om0
# must reproduce the from_lenstool conversion under the same background cosmology.
mp_default = al.mp.dPIEMassSph(
    sigma=sigma_lt,
    r_core=0.5,
    r_cut=40.0,
    redshift_object=z_l,
    redshift_source=z_s,
)
mp_flat = al.mp.dPIEMassB0Sph.from_lenstool(
    sigma=sigma_lt,
    r_core=0.5,
    r_cut=40.0,
    redshift_object=z_l,
    redshift_source=z_s,
    cosmology=ag.cosmo.FlatLambdaCDM(),
)
assert abs(mp_default.b0 / mp_flat.b0 - 1.0) < 1e-10
print("  default dPIEMassSph(sigma, r_core, r_cut) == dPIEMassB0Sph.from_lenstool  OK")

"""
__Leg 2: Spherical deflection vs convergence integral__

For a circular profile, alpha(R) = (2/R) * int_0^R kappa(r) r dr. The convergence and
deflection methods are independent ports, so quadrature agreement validates both.
"""
print("Leg 2: sph deflection vs convergence quadrature")

radii_check = np.array([0.3, 1.0, 3.0, 10.0, 60.0])
max_frac = 0.0
for R in radii_check:
    r_int = np.linspace(1e-6, R, 200001)
    kappa = mp_sph.convergence_2d_from(
        grid=ag.Grid2DIrregular(np.stack([np.zeros_like(r_int), r_int], axis=1))
    )
    alpha_quad = 2.0 / R * np.trapezoid(np.asarray(kappa) * r_int, r_int)
    alpha_direct = mp_sph.deflections_yx_2d_from(grid=ag.Grid2DIrregular([[0.0, R]]))[
        0, 1
    ]
    max_frac = max(max_frac, abs(alpha_quad / alpha_direct - 1.0))
assert max_frac < 1e-5
print_leg("alpha(R) vs quadrature", max_frac)

"""
__Leg 3: Analytic aperture "mass" (ra -> 0 closed form)__
"""
print("Leg 3: analytic enclosed-convergence anchor (ra -> 0)")

mp_zero_core = al.mp.dPIEMassB0Sph(centre=(0.0, 0.0), ra=1e-8, rs=25.0, b0=8.0)

max_frac = 0.0
for R in (0.5, 2.0, 10.0, 40.0):
    r_int = np.linspace(1e-7, R, 200001)
    kappa = mp_zero_core.convergence_2d_from(
        grid=ag.Grid2DIrregular(np.stack([np.zeros_like(r_int), r_int], axis=1))
    )
    integral = 2.0 * np.pi * np.trapezoid(np.asarray(kappa) * r_int, r_int)
    closed_form = np.pi * 8.0 * (R + 25.0 - np.sqrt(25.0**2 + R**2))
    max_frac = max(max_frac, abs(integral / closed_form - 1.0))
assert max_frac < 1e-4
print_leg("enclosed kappa vs closed form", max_frac)

"""
__Leg 4: Elliptical deflection vs potential gradient__
"""
print("Leg 4: elliptical deflections vs finite-difference potential gradient")

mp_ell = al.mp.dPIEMassB0.from_lenstool(
    ellipticity=0.4,
    angle_pos=20.0,
    sigma=800.0,
    r_core=1.5,
    r_cut=30.0,
    redshift_object=z_l,
    redshift_source=z_s,
    cosmology=cosmology,
)

eps_fd = 1e-6
points = [(0.8, 1.3), (-2.1, 0.4), (3.5, -2.9)]
max_frac = 0.0
for y, x in points:
    grid_fd = ag.Grid2DIrregular(
        [
            [y, x + eps_fd],
            [y, x - eps_fd],
            [y + eps_fd, x],
            [y - eps_fd, x],
        ]
    )
    psi = np.asarray(mp_ell.potential_2d_from(grid=grid_fd))
    alpha_fd_x = (psi[0] - psi[1]) / (2 * eps_fd)
    alpha_fd_y = (psi[2] - psi[3]) / (2 * eps_fd)
    alpha = mp_ell.deflections_yx_2d_from(grid=ag.Grid2DIrregular([[y, x]]))
    max_frac = max(
        max_frac,
        abs(alpha_fd_y / alpha[0, 0] - 1.0),
        abs(alpha_fd_x / alpha[0, 1] - 1.0),
    )
assert max_frac < 1e-4
print_leg("grad(psi) vs ci05f deflections", max_frac)

"""
__Leg 5: Hessian vs finite differences of the deflection field__

``analytical_hessian_2d_from`` evaluates in the profile frame (no transform decorator):
its components only equal world-frame deflection derivatives for angle = 0 profiles. The
determinant (magnification) is rotation-invariant, so downstream magnification use is
unaffected; this leg therefore tests an unrotated profile.
"""
print("Leg 5: analytical hessian vs deflection finite differences (profile frame)")

mp_ell_frame = al.mp.dPIEMassB0.from_lenstool(
    ellipticity=0.4,
    angle_pos=0.0,
    sigma=800.0,
    r_core=1.5,
    r_cut=30.0,
    redshift_object=z_l,
    redshift_source=z_s,
    cosmology=cosmology,
)

max_frac = 0.0
for y, x in points:
    grid_fd = ag.Grid2DIrregular(
        [
            [y, x + eps_fd],
            [y, x - eps_fd],
            [y + eps_fd, x],
            [y - eps_fd, x],
        ]
    )
    alpha_fd = np.asarray(mp_ell_frame.deflections_yx_2d_from(grid=grid_fd))
    h_xx_fd = (alpha_fd[0, 1] - alpha_fd[1, 1]) / (2 * eps_fd)
    h_yy_fd = (alpha_fd[2, 0] - alpha_fd[3, 0]) / (2 * eps_fd)
    h_xy_fd = (alpha_fd[2, 1] - alpha_fd[3, 1]) / (2 * eps_fd)

    h_yy, h_xy, h_yx, h_xx = mp_ell_frame.analytical_hessian_2d_from(
        grid=ag.Grid2DIrregular([[y, x]])
    )
    max_frac = max(
        max_frac,
        abs(h_xx_fd / float(h_xx[0]) - 1.0),
        abs(h_yy_fd / float(h_yy[0]) - 1.0),
        abs(h_xy_fd / float(h_xy[0]) - 1.0),
    )
assert max_frac < 1e-3
print_leg("hessian vs finite differences", max_frac)

"""
__Leg 6: Lenstool reference values through the from_lenstool path__

The hard-coded points below are the port's original validation values (generated with
Lenstool at porting time, see test_dual_pseudo_isothermal_mass.py). Re-derive the same
profile through from_lenstool by inverting b0 -> sigma_LT, so the whole conversion layer
sits in the loop.
"""
print("Leg 6: Lenstool port reference values via from_lenstool")

b0_target = 5.2
sigma_from_b0 = C_KM_S * np.sqrt(b0_target / (6.0 * 648000.0 * (d_ls / d_s)))

mp_ref = al.mp.dPIEMassB0Sph.from_lenstool(
    centre=(-0.7, 0.5),
    sigma=sigma_from_b0,
    r_core=2.0,
    r_cut=3.0,
    redshift_object=z_l,
    redshift_source=z_s,
    cosmology=cosmology,
)
assert abs(mp_ref.b0 / b0_target - 1.0) < 1e-10

deflections = mp_ref.deflections_yx_2d_from(grid=ag.Grid2DIrregular([[0.1875, 0.1625]]))
assert abs(deflections[0, 0] / 1.033080741 - 1.0) < 1e-4
assert abs(deflections[0, 1] / -0.39286169026 - 1.0) < 1e-4
print("  reference deflections reproduced through the conversion layer  OK")

print("\nAll Lenstool dPIE parity legs passed.")
