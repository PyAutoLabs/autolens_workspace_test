"""
Tracer: Multi-Plane Logic
=========================

This script tests the correctness of multi-plane ray-tracing logic in the `Tracer`
class, using only the NumPy path.  It is a companion to `tracer_jax.py`, which
verifies that the same calculations reproduce identical results on the JAX path and
under `jax.jit`.

No hardcoded numerical values are used as expected outputs.  All assertions are
relational: they test properties that must hold regardless of specific parameter
values (e.g. "with no mass the source-plane grid equals the image-plane grid").

__What is tested__

1.  No-mass tracer: source-only galaxy produces undeflected traced grids.
2.  Two-plane deflection: a lens at z=0.5 and source at z=1.0 actually deflects the
    source-plane grid away from the image-plane grid.
3.  Redshift order invariance: constructing a Tracer with galaxies listed in reverse
    redshift order gives the same traced grids as the forward-ordered Tracer.
4.  Coplanar additivity: two IsothermalSph profiles at the same plane with
    einstein_radius=R each produce deflections equal to one IsothermalSph with
    einstein_radius=2R (SIS deflections scale linearly with Einstein radius).
5.  Three-plane system: adding a second mass-bearing lens at z=1.0 in a
    lens(z=0.5) + lens(z=1.0) + source(z=2.0) system changes the source-plane grid
    compared to a two-plane system with only the z=0.5 lens.
6.  plane_index_limit early termination: requesting only the first N planes returns
    the same grids as the full calculation for those N planes.
7.  Plane grouping: two galaxies at the same redshift are grouped into one plane, so
    total_planes < len(galaxies).
"""

import numpy as np
import numpy.testing as npt
import autoarray as aa
import autogalaxy as ag
import autolens as al

"""
__Grids__

An irregular grid with well-separated, off-centre points is used for most tests.
A uniform grid is used where array-shape reasoning is needed (test 6).
"""
grid_irr = ag.Grid2DIrregular(values=[(0.5, 0.5), (1.0, -0.5), (-0.5, 1.0), (1.5, 1.5)])
grid_uni = ag.Grid2D.uniform(shape_native=(8, 8), pixel_scales=0.3)

"""
__Shared Profiles__

Reusable profile instances.  All mass profiles are centred at the origin and have
enough Einstein radius to produce clearly measurable deflections at the test-grid
points.
"""
isothermal_lens = ag.mp.IsothermalSph(centre=(0.0, 0.0), einstein_radius=1.0)
isothermal_lens_heavy = ag.mp.IsothermalSph(centre=(0.0, 0.0), einstein_radius=2.0)
sersic_source = ag.lp.Sersic(
    centre=(0.0, 0.0),
    ell_comps=(0.0, 0.0),
    intensity=0.1,
    effective_radius=0.3,
    sersic_index=1.0,
)

"""
__Test 1: No-mass tracer — undeflected grids__

A Tracer containing only a source galaxy (no mass profiles) must return traced grids
at every plane that are identical to the input image-plane grid.

Deflection angles of a massless galaxy are zero by definition, so the ray-tracing
formula reduces to the identity mapping at every plane.
"""
print("Test 1: No-mass tracer...")

source_only = al.Galaxy(redshift=1.0, bulge=sersic_source)
tracer_no_mass = al.Tracer(galaxies=[source_only])

traced_grids = tracer_no_mass.traced_grid_2d_list_from(grid=grid_irr)

assert (
    len(traced_grids) == 1
), f"No-mass tracer: expected 1 traced grid, got {len(traced_grids)}"

npt.assert_allclose(
    np.array(traced_grids[0]),
    np.array(grid_irr),
    rtol=1e-10,
    err_msg="No-mass tracer: traced grid does not equal the input grid",
)

print("  PASSED")

"""
__Test 2: Two-plane deflection — lensing actually deflects__

A lens at z=0.5 with a non-zero mass profile must produce a source-plane grid that
differs from the image-plane grid.

We check two things:
  - The returned list has two entries (one per plane).
  - The source-plane grid (index 1) is NOT equal to the image-plane grid (index 0),
    confirming that deflection angles have been applied.
"""
print("Test 2: Two-plane deflection...")

lens_galaxy = al.Galaxy(redshift=0.5, mass=isothermal_lens)
source_galaxy = al.Galaxy(redshift=1.0, bulge=sersic_source)
tracer_two_plane = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

traced_two = tracer_two_plane.traced_grid_2d_list_from(grid=grid_irr)

assert (
    len(traced_two) == 2
), f"Two-plane tracer: expected 2 traced grids, got {len(traced_two)}"

# Source-plane grid must differ from image-plane grid.
image_plane_coords = np.array(traced_two[0])
source_plane_coords = np.array(traced_two[1])

assert not np.allclose(
    image_plane_coords, source_plane_coords
), "Two-plane tracer: source-plane grid equals image-plane grid — no deflection occurred"

print("  PASSED")

"""
__Test 3: Redshift order invariance__

The Tracer sorts galaxies by ascending redshift internally before performing any
ray-tracing.  Constructing a Tracer with galaxies listed in reverse redshift order
must therefore yield traced grids identical to the forward-ordered Tracer.
"""
print("Test 3: Redshift order invariance...")

tracer_forward = al.Tracer(galaxies=[lens_galaxy, source_galaxy])
tracer_reversed = al.Tracer(galaxies=[source_galaxy, lens_galaxy])

traced_forward = tracer_forward.traced_grid_2d_list_from(grid=grid_irr)
traced_reversed = tracer_reversed.traced_grid_2d_list_from(grid=grid_irr)

assert len(traced_forward) == len(
    traced_reversed
), "Redshift order invariance: different number of planes"

for i, (g_fwd, g_rev) in enumerate(zip(traced_forward, traced_reversed)):
    npt.assert_allclose(
        np.array(g_fwd),
        np.array(g_rev),
        rtol=1e-10,
        err_msg=f"Redshift order invariance: plane {i} mismatch between forward and reversed tracer",
    )

print("  PASSED")

"""
__Test 4: Coplanar additivity__

For a Singular Isothermal Sphere (SIS), deflection angles scale linearly with the
Einstein radius:

    alpha(R) = R * (direction unit vector)

Therefore two co-located SIS profiles each with Einstein radius R must produce
the same total deflections as one SIS with Einstein radius 2R.

We verify this by comparing:
  - A Tracer with two SIS profiles (einstein_radius=1.0 each) at z=0.5.
  - A Tracer with one SIS profile (einstein_radius=2.0) at z=0.5.
Both have an identical source galaxy at z=1.0.

The source-plane traced grids must be identical.
"""
print("Test 4: Coplanar additivity (two IsothermalSph = one with double radius)...")

lens_two_sph = al.Galaxy(
    redshift=0.5,
    mass_0=ag.mp.IsothermalSph(centre=(0.0, 0.0), einstein_radius=1.0),
    mass_1=ag.mp.IsothermalSph(centre=(0.0, 0.0), einstein_radius=1.0),
)
lens_one_heavy = al.Galaxy(
    redshift=0.5,
    mass=ag.mp.IsothermalSph(centre=(0.0, 0.0), einstein_radius=2.0),
)
source_galaxy_plain = al.Galaxy(redshift=1.0)

tracer_two_sph = al.Tracer(galaxies=[lens_two_sph, source_galaxy_plain])
tracer_one_heavy = al.Tracer(galaxies=[lens_one_heavy, source_galaxy_plain])

traced_two_sph = tracer_two_sph.traced_grid_2d_list_from(grid=grid_irr)
traced_one_heavy = tracer_one_heavy.traced_grid_2d_list_from(grid=grid_irr)

npt.assert_allclose(
    np.array(traced_two_sph[-1]),
    np.array(traced_one_heavy[-1]),
    rtol=1e-6,
    err_msg="Coplanar additivity: two IsothermalSph(R=1) != one IsothermalSph(R=2)",
)

print("  PASSED")

"""
__Test 5: Three-plane system — intermediate lens changes source-plane grid__

Adding a second mass-bearing lens at z=1.0 to a two-plane system must change the
source-plane (z=2.0) grid.

Concretely we compare:
  - tracer_2p: lens(z=0.5) + source(z=2.0) — two planes.
  - tracer_3p: lens(z=0.5) + lens2(z=1.0) + source(z=2.0) — three planes.

Assertions:
  - tracer_3p returns 3 traced grids.
  - The first plane grid (z=0.5) is identical between the two tracers (the
    z=0.5 plane sees no deflections from planes further away).
  - The source-plane grid (z=2.0) differs between the two tracers (the second
    lens at z=1.0 adds extra deflection).
"""
print("Test 5: Three-plane system...")

lens_z05 = al.Galaxy(redshift=0.5, mass=isothermal_lens)
lens_z10 = al.Galaxy(redshift=1.0, mass=isothermal_lens)
source_z20 = al.Galaxy(redshift=2.0)

tracer_2p = al.Tracer(galaxies=[lens_z05, source_z20])
tracer_3p = al.Tracer(galaxies=[lens_z05, lens_z10, source_z20])

traced_2p = tracer_2p.traced_grid_2d_list_from(grid=grid_irr)
traced_3p = tracer_3p.traced_grid_2d_list_from(grid=grid_irr)

assert (
    len(traced_3p) == 3
), f"Three-plane tracer: expected 3 planes, got {len(traced_3p)}"

# First plane (z=0.5) should be the same: no upstream mass deflects light before
# reaching z=0.5 in either tracer.
npt.assert_allclose(
    np.array(traced_2p[0]),
    np.array(traced_3p[0]),
    rtol=1e-10,
    err_msg="Three-plane test: plane 0 (z=0.5) differs between 2p and 3p tracer",
)

# Source plane (last) must differ: the extra lens at z=1.0 adds deflection.
assert not np.allclose(
    np.array(traced_2p[-1]), np.array(traced_3p[-1])
), "Three-plane test: source-plane grids are equal — second lens had no effect"

print("  PASSED")

"""
__Test 6: plane_index_limit early termination__

With `plane_index_limit=1` on a three-plane tracer, only the first two planes
(indexes 0 and 1) should be returned.  The two returned grids must match the first
two grids of the full three-plane calculation because terminating early does not
affect the lower-plane calculations.
"""
print("Test 6: plane_index_limit early termination...")

tracer_3p_full = al.Tracer(galaxies=[lens_z05, lens_z10, source_z20])

traced_full = tracer_3p_full.traced_grid_2d_list_from(grid=grid_irr)
traced_limited = tracer_3p_full.traced_grid_2d_list_from(
    grid=grid_irr, plane_index_limit=1
)

assert (
    len(traced_limited) == 2
), f"plane_index_limit=1: expected 2 grids, got {len(traced_limited)}"

for i in range(2):
    npt.assert_allclose(
        np.array(traced_limited[i]),
        np.array(traced_full[i]),
        rtol=1e-10,
        err_msg=f"plane_index_limit: plane {i} differs between limited and full calculation",
    )

print("  PASSED")

"""
__Test 7: Plane grouping — co-redshift galaxies share a plane__

When two galaxies share the same redshift they must be grouped into the same plane.
The resulting Tracer must therefore have fewer planes than galaxies.

We also verify that `total_planes` matches the length of the `planes` list.
"""
print("Test 7: Plane grouping...")

gal_a = al.Galaxy(redshift=0.5, mass=isothermal_lens)
gal_b = al.Galaxy(
    redshift=0.5, mass=ag.mp.IsothermalSph(centre=(0.5, 0.5), einstein_radius=0.5)
)
gal_c = al.Galaxy(redshift=1.0)

tracer_grouped = al.Tracer(galaxies=[gal_a, gal_b, gal_c])

assert (
    tracer_grouped.total_planes == 2
), f"Plane grouping: expected 2 planes, got {tracer_grouped.total_planes}"
assert tracer_grouped.total_planes == len(
    tracer_grouped.planes
), "Plane grouping: total_planes != len(planes)"
assert (
    len(tracer_grouped.galaxies) == 3
), "Plane grouping: galaxy count changed unexpectedly"

# The first plane must contain both co-redshift galaxies.
assert (
    len(tracer_grouped.planes[0]) == 2
), f"Plane grouping: expected 2 galaxies in plane 0, got {len(tracer_grouped.planes[0])}"

print("  PASSED")

print("\nAll tracer_multiplane.py checks passed.")
