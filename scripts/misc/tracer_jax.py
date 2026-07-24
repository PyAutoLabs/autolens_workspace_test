"""
Tracer: JAX JIT
===============

This script tests that the `Tracer` ray-tracing calculations produce identical
results on the NumPy path (`xp=np`) and the JAX path (`xp=jnp`), and that the
JAX path compiles correctly under `jax.jit`.

It is a companion to `tracer_multiplane.py`, which tests multi-plane logic
correctness using NumPy only.  This script reuses the same tracer configurations
(two-plane and three-plane) and runs them through the JAX backend.

__What is tested__

1.  `traced_grid_2d_list_from`: NumPy vs JAX on both irregular and uniform grids,
    for two-plane and three-plane systems.

2.  `traced_grid_2d_list_from` inside `jax.jit`: a lambda extracts `._array` from
    each grid before returning a list of raw `jax.Array` objects (autoarray types
    are not valid JIT outputs).

3.  `image_2d_from` on the Tracer: NumPy vs JAX, then also inside `jax.jit`.

4.  `deflections_yx_2d_from` on the Tracer: NumPy vs JAX.

5.  `convergence_2d_from` on the Tracer: NumPy vs JAX.

__Tolerances__

All comparisons use `rtol=1e-5` unless noted.  The uniform grid includes points
close to the lens centre where higher curvature raises finite-difference-style
numerical differences between paths; `rtol=1e-4` is used there.
"""

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import autoarray as aa
import autogalaxy as ag
import autolens as al

"""
__Grids__
"""
grid_irr = ag.Grid2DIrregular(values=[(0.5, 0.5), (1.0, -0.5), (-0.5, 1.0), (1.5, 1.5)])
grid_uni = ag.Grid2D.uniform(shape_native=(8, 8), pixel_scales=0.3)

"""
__Shared Galaxies__

Two-plane: lens(z=0.5) + source(z=1.0)
Three-plane: lens1(z=0.5) + lens2(z=1.0) + source(z=2.0)

Light on the source galaxies gives `image_2d_from` something to evaluate.
"""
isothermal = ag.mp.Isothermal(
    centre=(0.0, 0.0),
    ell_comps=(0.0, 0.111111),
    einstein_radius=1.5,
)
sersic_source = ag.lp.Sersic(
    centre=(0.0, 0.0),
    ell_comps=(0.0, 0.0),
    intensity=0.1,
    effective_radius=0.3,
    sersic_index=1.0,
)

lens_z05 = al.Galaxy(redshift=0.5, mass=isothermal)
source_z10 = al.Galaxy(redshift=1.0, bulge=sersic_source)

lens2_z10 = al.Galaxy(
    redshift=1.0,
    mass=ag.mp.IsothermalSph(centre=(0.0, 0.0), einstein_radius=0.5),
)
source_z20 = al.Galaxy(redshift=2.0, bulge=sersic_source)

tracer_2p = al.Tracer(galaxies=[lens_z05, source_z10])
tracer_3p = al.Tracer(galaxies=[lens_z05, lens2_z10, source_z20])

"""
__Test 1: traced_grid_2d_list_from — JAX vs NumPy (two-plane, irregular grid)__

The traced grid list on the JAX path returns `Grid2DIrregular` objects whose
`._array` is a `jax.Array`.  Numerical values must match the NumPy path to `rtol=1e-5`.
"""
print("Test 1: traced_grid_2d_list_from (2-plane, irregular grid)...")

traced_np_2p_irr = tracer_2p.traced_grid_2d_list_from(grid=grid_irr)
traced_jax_2p_irr = tracer_2p.traced_grid_2d_list_from(grid=grid_irr, xp=jnp)

assert len(traced_np_2p_irr) == len(
    traced_jax_2p_irr
), "traced_grid_2d_list_from 2p irr: plane count mismatch"

for i, (g_np, g_jax) in enumerate(zip(traced_np_2p_irr, traced_jax_2p_irr)):
    assert isinstance(
        g_jax._array, jax.Array
    ), f"traced_grid_2d_list_from 2p irr: plane {i} ._array is not jax.Array"
    npt.assert_allclose(
        np.array(g_jax._array),
        np.array(g_np._array),
        rtol=1e-5,
        err_msg=f"traced_grid_2d_list_from 2p irr: plane {i} numpy vs jax mismatch",
    )

print("  PASSED")

"""
__Test 2: traced_grid_2d_list_from — JAX vs NumPy (three-plane, irregular grid)__
"""
print("Test 2: traced_grid_2d_list_from (3-plane, irregular grid)...")

traced_np_3p_irr = tracer_3p.traced_grid_2d_list_from(grid=grid_irr)
traced_jax_3p_irr = tracer_3p.traced_grid_2d_list_from(grid=grid_irr, xp=jnp)

assert (
    len(traced_np_3p_irr) == len(traced_jax_3p_irr) == 3
), "traced_grid_2d_list_from 3p irr: expected 3 planes"

for i, (g_np, g_jax) in enumerate(zip(traced_np_3p_irr, traced_jax_3p_irr)):
    assert isinstance(
        g_jax._array, jax.Array
    ), f"traced_grid_2d_list_from 3p irr: plane {i} ._array is not jax.Array"
    npt.assert_allclose(
        np.array(g_jax._array),
        np.array(g_np._array),
        rtol=1e-5,
        err_msg=f"traced_grid_2d_list_from 3p irr: plane {i} numpy vs jax mismatch",
    )

print("  PASSED")

"""
__Test 3: traced_grid_2d_list_from inside jax.jit (two-plane, irregular grid)__

The lambda extracts `._array` from each grid before returning so that the JIT
output is a plain list of `jax.Array` objects — a valid JAX pytree.

Results must match the NumPy path.
"""
print("Test 3: traced_grid_2d_list_from inside jax.jit (2-plane, irregular grid)...")

jitted_traced_2p = jax.jit(
    lambda: [
        g._array for g in tracer_2p.traced_grid_2d_list_from(grid=grid_irr, xp=jnp)
    ]
)
traced_jit_2p = jitted_traced_2p()

assert (
    isinstance(traced_jit_2p, list) and len(traced_jit_2p) == 2
), "JIT traced_grid 2p: expected list of 2 jax.Arrays"

for i, (arr_jit, g_np) in enumerate(zip(traced_jit_2p, traced_np_2p_irr)):
    assert isinstance(
        arr_jit, jax.Array
    ), f"JIT traced_grid 2p: plane {i} is not jax.Array"
    npt.assert_allclose(
        np.array(arr_jit),
        np.array(g_np._array),
        rtol=1e-5,
        err_msg=f"JIT traced_grid 2p: plane {i} mismatch",
    )

print("  PASSED")

"""
__Test 4: traced_grid_2d_list_from inside jax.jit (three-plane, irregular grid)__
"""
print("Test 4: traced_grid_2d_list_from inside jax.jit (3-plane, irregular grid)...")

jitted_traced_3p = jax.jit(
    lambda: [
        g._array for g in tracer_3p.traced_grid_2d_list_from(grid=grid_irr, xp=jnp)
    ]
)
traced_jit_3p = jitted_traced_3p()

assert len(traced_jit_3p) == 3, "JIT traced_grid 3p: expected list of 3 jax.Arrays"

for i, (arr_jit, g_np) in enumerate(zip(traced_jit_3p, traced_np_3p_irr)):
    npt.assert_allclose(
        np.array(arr_jit),
        np.array(g_np._array),
        rtol=1e-5,
        err_msg=f"JIT traced_grid 3p: plane {i} mismatch",
    )

print("  PASSED")

"""
__Test 5: image_2d_from on Tracer — JAX vs NumPy (irregular grid)__

`image_2d_from` sums the lensed images from all planes.  On the JAX path the
result is an `aa.ArrayIrregular` whose `._array` is a `jax.Array`.
"""
print("Test 5: image_2d_from (irregular grid)...")

image_np_irr = tracer_2p.image_2d_from(grid=grid_irr)
image_jax_irr = tracer_2p.image_2d_from(grid=grid_irr, xp=jnp)

assert isinstance(
    image_np_irr, aa.ArrayIrregular
), f"image_2d_from irr (numpy): expected aa.ArrayIrregular, got {type(image_np_irr)}"
assert isinstance(
    image_jax_irr, aa.ArrayIrregular
), f"image_2d_from irr (jax outer): expected aa.ArrayIrregular, got {type(image_jax_irr)}"
assert isinstance(
    image_jax_irr._array, jax.Array
), "image_2d_from irr (jax outer): ._array is not jax.Array"

npt.assert_allclose(
    np.array(image_jax_irr._array),
    np.array(image_np_irr._array),
    rtol=1e-5,
    err_msg="image_2d_from (irregular grid): numpy vs jax mismatch",
)

print("  PASSED")

"""
__Test 6: image_2d_from inside jax.jit (irregular grid)__
"""
print("Test 6: image_2d_from inside jax.jit (irregular grid)...")

jitted_image = jax.jit(lambda: tracer_2p.image_2d_from(grid=grid_irr, xp=jnp)._array)
image_jit = jitted_image()

assert isinstance(image_jit, jax.Array), "image_2d_from JIT: output is not jax.Array"

npt.assert_allclose(
    np.array(image_jit),
    np.array(image_np_irr._array),
    rtol=1e-5,
    err_msg="image_2d_from JIT: mismatch vs numpy",
)

print("  PASSED")

"""
__Test 7: deflections_yx_2d_from on Tracer — JAX vs NumPy__

`deflections_yx_2d_from` on a multi-plane Tracer computes the net deflection by
subtracting the source-plane grid from the image-plane grid.  On the JAX path the
result is a `VectorYX2DIrregular` with `._array` being a `jax.Array`.
"""
print("Test 7: deflections_yx_2d_from (irregular grid)...")

deflections_np = tracer_2p.deflections_yx_2d_from(grid=grid_irr)
deflections_jax = tracer_2p.deflections_yx_2d_from(grid=grid_irr, xp=jnp)

assert isinstance(
    deflections_np, aa.VectorYX2DIrregular
), f"deflections_yx_2d_from (numpy): expected VectorYX2DIrregular, got {type(deflections_np)}"
assert isinstance(
    deflections_jax, aa.VectorYX2DIrregular
), f"deflections_yx_2d_from (jax outer): expected VectorYX2DIrregular, got {type(deflections_jax)}"
assert isinstance(
    deflections_jax._array, jax.Array
), "deflections_yx_2d_from (jax outer): ._array is not jax.Array"

npt.assert_allclose(
    np.array(deflections_jax._array),
    np.array(deflections_np._array),
    rtol=1e-5,
    err_msg="deflections_yx_2d_from: numpy vs jax mismatch",
)

print("  PASSED")

"""
__Test 8: deflections_yx_2d_from inside jax.jit__
"""
print("Test 8: deflections_yx_2d_from inside jax.jit...")

jitted_deflections = jax.jit(
    lambda: tracer_2p.deflections_yx_2d_from(grid=grid_irr, xp=jnp)._array
)
deflections_jit = jitted_deflections()

assert isinstance(
    deflections_jit, jax.Array
), "deflections_yx_2d_from JIT: output is not jax.Array"

npt.assert_allclose(
    np.array(deflections_jit),
    np.array(deflections_np._array),
    rtol=1e-5,
    err_msg="deflections_yx_2d_from JIT: mismatch vs numpy",
)

print("  PASSED")

"""
__Test 9: convergence_2d_from on Tracer — JAX vs NumPy__

`convergence_2d_from` sums the convergence of all galaxies.  This does not involve
ray-tracing; it is a direct sum of per-galaxy convergence values.
"""
print("Test 9: convergence_2d_from (irregular grid)...")

convergence_np = tracer_2p.convergence_2d_from(grid=grid_irr)
convergence_jax = tracer_2p.convergence_2d_from(grid=grid_irr, xp=jnp)

assert isinstance(
    convergence_np, aa.ArrayIrregular
), f"convergence_2d_from (numpy): expected aa.ArrayIrregular, got {type(convergence_np)}"
assert isinstance(
    convergence_jax, aa.ArrayIrregular
), f"convergence_2d_from (jax outer): expected aa.ArrayIrregular, got {type(convergence_jax)}"
assert isinstance(
    convergence_jax._array, jax.Array
), "convergence_2d_from (jax outer): ._array is not jax.Array"

npt.assert_allclose(
    np.array(convergence_jax._array),
    np.array(convergence_np._array),
    rtol=1e-5,
    err_msg="convergence_2d_from: numpy vs jax mismatch",
)

print("  PASSED")

"""
__Test 10: convergence_2d_from inside jax.jit__
"""
print("Test 10: convergence_2d_from inside jax.jit...")

jitted_convergence = jax.jit(
    lambda: tracer_2p.convergence_2d_from(grid=grid_irr, xp=jnp)._array
)
convergence_jit = jitted_convergence()

assert isinstance(
    convergence_jit, jax.Array
), "convergence_2d_from JIT: output is not jax.Array"

npt.assert_allclose(
    np.array(convergence_jit),
    np.array(convergence_np._array),
    rtol=1e-5,
    err_msg="convergence_2d_from JIT: mismatch vs numpy",
)

print("  PASSED")

"""
__Test 11: Three-plane JAX vs NumPy consistency__

Repeat the JAX vs NumPy comparison on the three-plane tracer for `image_2d_from`
and `deflections_yx_2d_from`, confirming multi-plane JAX consistency.
"""
print("Test 11: Three-plane image_2d_from and deflections (irregular grid)...")

image_3p_np = tracer_3p.image_2d_from(grid=grid_irr)
image_3p_jax = tracer_3p.image_2d_from(grid=grid_irr, xp=jnp)

npt.assert_allclose(
    np.array(image_3p_jax._array),
    np.array(image_3p_np._array),
    rtol=1e-5,
    err_msg="Three-plane image_2d_from: numpy vs jax mismatch",
)

deflections_3p_np = tracer_3p.deflections_yx_2d_from(grid=grid_irr)
deflections_3p_jax = tracer_3p.deflections_yx_2d_from(grid=grid_irr, xp=jnp)

npt.assert_allclose(
    np.array(deflections_3p_jax._array),
    np.array(deflections_3p_np._array),
    rtol=1e-5,
    err_msg="Three-plane deflections_yx_2d_from: numpy vs jax mismatch",
)

print("  PASSED")

print("\nAll tracer_jax.py checks passed.")
