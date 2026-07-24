"""
Operate: Hessian JAX vs NumPy
==============================

This script compares the NumPy and JAX computational paths for `hessian_from` and all
functions that depend on it, verifying they produce consistent results and return the
expected types.

The NumPy path uses a finite-difference approximation (the default, `xp=np`).
The JAX path uses exact derivatives via `jax.jacfwd` (`xp=jnp`), wrapped in `jax.jit`.

Comparisons use `numpy.testing.assert_allclose`. For points well away from the profile
centre a tolerance of `rtol=1e-3` is used; for a uniform grid that includes points
near the centre (where higher curvature raises finite-difference error) `rtol=5e-3`
is used.
"""

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import autoarray as aa

import autogalaxy as ag
from autogalaxy.operate.lens_calc import LensCalc

"""
__Mass Profile, LensCalc, and Grids__

We use an `Isothermal` mass profile as the test object, evaluated on both an irregular
grid and a uniform grid to exercise both code paths in `deflections_yx_scalar`.

`LensCalc.from_mass_obj(mp)` wraps the profile's `deflections_yx_2d_from`
callable and exposes all hessian-derived lensing operations.
"""
mp = ag.mp.Isothermal(
    centre=(0.0, 0.0), ell_comps=(0.0, -0.111111), einstein_radius=2.0
)

lens_calc = LensCalc.from_mass_obj(mp)

grid_irregular = ag.Grid2DIrregular(values=[(0.5, 0.5), (1.0, 1.0)])
grid_uniform = ag.Grid2D.uniform(shape_native=(10, 10), pixel_scales=0.3)

"""
__deflections_yx_2d_from (Irregular Grid)__

`deflections_yx_2d_from` does not use the hessian module; it computes deflection angles
directly from the mass profile analytic formula.  It is still called directly on `mp`
because `deflections_yx_2d_from` is a `MassProfile` method, not an `LensCalc`
method.

The `@aa.grid_dec.to_vector_yx` decorator always wraps the result in an autoarray type
regardless of which backend is used:
  - NumPy path (`xp=np`)  → `aa.VectorYX2DIrregular`; internal `._array` is `np.ndarray`
  - JAX path   (`xp=jnp`) → `aa.VectorYX2DIrregular`; internal `._array` is `jax.Array`

In both cases the outer object is an autoarray structure, *not* a raw `jax.Array`.
The JAX-ness is carried by the `._array` attribute.  This contrasts with functions
in `LensCalc` such as `convergence_2d_via_hessian_from`, which guard their autoarray
wrapping with `if xp is np:` and return a bare `jax.Array` on the JAX path so that
they can be used safely inside `jax.jit`.
"""
deflections_np_irr = mp.deflections_yx_2d_from(grid=grid_irregular)

assert isinstance(deflections_np_irr, aa.VectorYX2DIrregular), (
    f"deflections_yx_2d_from (numpy, irregular): expected aa.VectorYX2DIrregular, "
    f"got {type(deflections_np_irr)}"
)
assert isinstance(deflections_np_irr._array, np.ndarray), (
    f"deflections_yx_2d_from (numpy, irregular): expected ._array to be np.ndarray, "
    f"got {type(deflections_np_irr._array)}"
)

deflections_jax_irr = mp.deflections_yx_2d_from(grid=grid_irregular, xp=jnp)

assert isinstance(deflections_jax_irr, aa.VectorYX2DIrregular), (
    f"deflections_yx_2d_from (jax, irregular): expected aa.VectorYX2DIrregular, "
    f"got {type(deflections_jax_irr)}"
)
assert isinstance(deflections_jax_irr._array, jax.Array), (
    f"deflections_yx_2d_from (jax, irregular): expected ._array to be jax.Array, "
    f"got {type(deflections_jax_irr._array)}"
)

npt.assert_allclose(
    np.array(deflections_jax_irr._array),
    np.array(deflections_np_irr._array),
    rtol=1e-6,
    err_msg="deflections_yx_2d_from (irregular grid): numpy vs jax mismatch",
)

"""
__deflections_yx_2d_from (Uniform Grid)__

On a `Grid2D` (uniform) the decorator wraps the result in `aa.VectorYX2D` instead of
`aa.VectorYX2DIrregular`.  The type of `._array` follows the same rule as above.
"""
deflections_np_uni = mp.deflections_yx_2d_from(grid=grid_uniform)

assert isinstance(deflections_np_uni, aa.VectorYX2D), (
    f"deflections_yx_2d_from (numpy, uniform): expected aa.VectorYX2D, "
    f"got {type(deflections_np_uni)}"
)
assert isinstance(deflections_np_uni._array, np.ndarray), (
    f"deflections_yx_2d_from (numpy, uniform): expected ._array to be np.ndarray, "
    f"got {type(deflections_np_uni._array)}"
)

deflections_jax_uni = mp.deflections_yx_2d_from(grid=grid_uniform, xp=jnp)

assert isinstance(deflections_jax_uni, aa.VectorYX2D), (
    f"deflections_yx_2d_from (jax, uniform): expected aa.VectorYX2D, "
    f"got {type(deflections_jax_uni)}"
)
assert isinstance(deflections_jax_uni._array, jax.Array), (
    f"deflections_yx_2d_from (jax, uniform): expected ._array to be jax.Array, "
    f"got {type(deflections_jax_uni._array)}"
)

npt.assert_allclose(
    np.array(deflections_jax_uni._array),
    np.array(deflections_np_uni._array),
    rtol=1e-6,
    err_msg="deflections_yx_2d_from (uniform grid): numpy vs jax mismatch",
)

"""
__hessian_from (Irregular Grid)__

Compare the four Hessian components on an irregular grid, where `pixel_scales` is not
available and the JAX path falls back to the default `(0.05, 0.05)`.

The NumPy path returns a tuple of four plain `numpy.ndarray` components.
The JAX path returns a tuple of four `jax.Array` components.
"""
hessian_np = lens_calc.hessian_from(grid=grid_irregular)

assert isinstance(hessian_np, tuple), "hessian_from (numpy): expected tuple"
assert len(hessian_np) == 4, "hessian_from (numpy): expected 4 components"
for component, name in zip(
    hessian_np, ("hessian_yy", "hessian_xy", "hessian_yx", "hessian_xx")
):
    assert isinstance(
        component, np.ndarray
    ), f"hessian_from (numpy, irregular grid): {name} expected np.ndarray, got {type(component)}"

hessian_jax_fn = jax.jit(lambda: lens_calc.hessian_from(grid=grid_irregular, xp=jnp))
hessian_jax = hessian_jax_fn()

assert isinstance(hessian_jax, tuple), "hessian_from (jax): expected tuple"
assert len(hessian_jax) == 4, "hessian_from (jax): expected 4 components"
for component, name in zip(
    hessian_jax, ("hessian_yy", "hessian_xy", "hessian_yx", "hessian_xx")
):
    assert isinstance(
        component, jax.Array
    ), f"hessian_from (jax, irregular grid): {name} expected jax.Array, got {type(component)}"

for component_np, component_jax, name in zip(
    hessian_np,
    hessian_jax,
    ("hessian_yy", "hessian_xy", "hessian_yx", "hessian_xx"),
):
    npt.assert_allclose(
        np.array(component_jax),
        np.array(component_np),
        rtol=1e-3,
        err_msg=f"hessian_from (irregular grid): mismatch in {name}",
    )

"""
__hessian_from (Uniform Grid)__

Repeat on a uniform `Grid2D` so that `pixel_scales` is read from the grid and passed
through to the internal `(1, 1)` mask used by `deflections_yx_scalar`.

A looser tolerance (`rtol=5e-3`) is used here because some grid points fall close to
the profile centre where the Isothermal Hessian has high curvature, raising the
finite-difference truncation error above the typical `~1e-4` level.
"""
hessian_np_uniform = lens_calc.hessian_from(grid=grid_uniform)

assert isinstance(
    hessian_np_uniform, tuple
), "hessian_from uniform (numpy): expected tuple"
for component, name in zip(
    hessian_np_uniform, ("hessian_yy", "hessian_xy", "hessian_yx", "hessian_xx")
):
    assert isinstance(
        component, np.ndarray
    ), f"hessian_from (numpy, uniform grid): {name} expected np.ndarray, got {type(component)}"

hessian_jax_uniform_fn = jax.jit(
    lambda: lens_calc.hessian_from(grid=grid_uniform, xp=jnp)
)
hessian_jax_uniform = hessian_jax_uniform_fn()

assert isinstance(
    hessian_jax_uniform, tuple
), "hessian_from uniform (jax): expected tuple"
for component, name in zip(
    hessian_jax_uniform, ("hessian_yy", "hessian_xy", "hessian_yx", "hessian_xx")
):
    assert isinstance(
        component, jax.Array
    ), f"hessian_from (jax, uniform grid): {name} expected jax.Array, got {type(component)}"

for component_np, component_jax, name in zip(
    hessian_np_uniform,
    hessian_jax_uniform,
    ("hessian_yy", "hessian_xy", "hessian_yx", "hessian_xx"),
):
    npt.assert_allclose(
        np.array(component_jax),
        np.array(component_np),
        rtol=5e-3,
        err_msg=f"hessian_from (uniform grid): mismatch in {name}",
    )

"""
__convergence_2d_via_hessian_from__

The convergence is `0.5 * (hessian_yy + hessian_xx)`.

The `xp` parameter is now threaded through into `hessian_from`, so both backends are
supported:
  - `xp=np`  → returns `aa.ArrayIrregular`
  - `xp=jnp` → returns a raw `jax.Array` (the `@to_array` decorator skips wrapping)
"""
convergence_np = lens_calc.convergence_2d_via_hessian_from(grid=grid_irregular)

assert isinstance(
    convergence_np, aa.ArrayIrregular
), f"convergence_2d_via_hessian_from (numpy): expected aa.ArrayIrregular, got {type(convergence_np)}"

convergence_jax_fn = jax.jit(
    lambda: lens_calc.convergence_2d_via_hessian_from(grid=grid_irregular, xp=jnp)
)
convergence_jax = convergence_jax_fn()

assert isinstance(
    convergence_jax, jax.Array
), f"convergence_2d_via_hessian_from (jax): expected jax.Array, got {type(convergence_jax)}"

npt.assert_allclose(
    np.array(convergence_jax),
    np.array(convergence_np),
    rtol=1e-3,
    err_msg="convergence_2d_via_hessian_from: numpy vs jax mismatch",
)

"""
__shear_yx_2d_via_hessian_from__

The shear components `gamma_1` and `gamma_2` are derived from the off-diagonal and
diagonal Hessian components respectively.

The `xp` parameter is threaded through into `hessian_from`:
  - `xp=np`  → returns `ag.ShearYX2DIrregular` (shape `(N, 2)`, col 0 = gamma_2,
    col 1 = gamma_1)
  - `xp=jnp` → returns a raw `jax.Array` of shape `(N, 2)` with the same column
    ordering (`ShearYX2DIrregular` wrapping is guarded with `if xp is np:`)
"""
shear_np = lens_calc.shear_yx_2d_via_hessian_from(grid=grid_irregular)

assert isinstance(
    shear_np, ag.ShearYX2DIrregular
), f"shear_yx_2d_via_hessian_from (numpy): expected ag.ShearYX2DIrregular, got {type(shear_np)}"

shear_jax_fn = jax.jit(
    lambda: lens_calc.shear_yx_2d_via_hessian_from(grid=grid_irregular, xp=jnp)
)
shear_jax = shear_jax_fn()

assert isinstance(
    shear_jax, jax.Array
), f"shear_yx_2d_via_hessian_from (jax): expected jax.Array, got {type(shear_jax)}"

npt.assert_allclose(
    np.array(shear_jax),
    np.array(shear_np),
    rtol=1e-3,
    atol=1e-3,
    err_msg="shear_yx_2d_via_hessian_from: numpy vs jax mismatch",
)

"""
__magnification_2d_via_hessian_from__

The magnification is `1 / det(A)` where `A` is the lensing Jacobian expressed via the
Hessian components. The `xp` parameter is threaded through from this function into
`hessian_from`.

When `xp=np` the result is an `aa.ArrayIrregular`. When `xp=jnp` the result is a raw
`jax.Array` because the `@to_array` decorator skips autoarray wrapping for the JAX path.
"""
mag_np = lens_calc.magnification_2d_via_hessian_from(grid=grid_irregular)

assert isinstance(
    mag_np, aa.ArrayIrregular
), f"magnification_2d_via_hessian_from (numpy): expected aa.ArrayIrregular, got {type(mag_np)}"

mag_jax_fn = jax.jit(
    lambda: lens_calc.magnification_2d_via_hessian_from(grid=grid_irregular, xp=jnp)
)
mag_jax = mag_jax_fn()

assert isinstance(
    mag_jax, jax.Array
), f"magnification_2d_via_hessian_from (jax): expected jax.Array, got {type(mag_jax)}"

npt.assert_allclose(
    np.array(mag_jax),
    np.array(mag_np),
    rtol=1e-3,
    err_msg="magnification_2d_via_hessian_from: mismatch",
)

"""
__jacobian_from__

The lensing Jacobian is `A = I - H`, where `H` is the Hessian of the deflection angles.
`jacobian_from` returns a 2x2 list of lists `[[a11, a12], [a21, a22]]` whose components
are plain `numpy.ndarray` (NumPy path) or `jax.Array` (JAX path).

The NumPy and JAX paths must agree on all four components to within the same tolerance
as `hessian_from` itself, since the Jacobian is a simple linear transformation of the
Hessian.
"""
jacobian_np = lens_calc.jacobian_from(grid=grid_irregular)

assert (
    isinstance(jacobian_np, list) and len(jacobian_np) == 2
), "jacobian_from (numpy): expected list of length 2"
assert (
    len(jacobian_np[0]) == 2 and len(jacobian_np[1]) == 2
), "jacobian_from (numpy): each row must have 2 elements"
for i, j, name in ((0, 0, "a11"), (0, 1, "a12"), (1, 0, "a21"), (1, 1, "a22")):
    assert isinstance(
        jacobian_np[i][j], np.ndarray
    ), f"jacobian_from (numpy): {name} expected np.ndarray, got {type(jacobian_np[i][j])}"

jacobian_jax_fn = jax.jit(lambda: lens_calc.jacobian_from(grid=grid_irregular, xp=jnp))
jacobian_jax = jacobian_jax_fn()

assert (
    isinstance(jacobian_jax, list) and len(jacobian_jax) == 2
), "jacobian_from (jax): expected list of length 2"
for i, j, name in ((0, 0, "a11"), (0, 1, "a12"), (1, 0, "a21"), (1, 1, "a22")):
    assert isinstance(
        jacobian_jax[i][j], jax.Array
    ), f"jacobian_from (jax): {name} expected jax.Array, got {type(jacobian_jax[i][j])}"

for i, j, name in ((0, 0, "a11"), (0, 1, "a12"), (1, 0, "a21"), (1, 1, "a22")):
    npt.assert_allclose(
        np.array(jacobian_jax[i][j]),
        np.array(jacobian_np[i][j]),
        rtol=1e-3,
        err_msg=f"jacobian_from (irregular grid): mismatch in {name}",
    )

"""
__tangential_eigen_value_from__

The tangential eigenvalue is `1 - convergence - |shear|`, where both convergence and
shear are derived from the Hessian via `convergence_2d_via_hessian_from` and
`shear_yx_2d_via_hessian_from`.

These functions use `aa.Array2D(mask=grid.mask)` on the NumPy return path, so they
require a uniform `Grid2D` (which carries a `mask`).

  - `xp=np`  → returns `aa.Array2D`
  - `xp=jnp` → returns a raw `jax.Array`; the `@to_array` decorator skips wrapping,
    and shear magnitudes are computed from the raw `(N, 2)` array as `sqrt(col0²+col1²)`
"""
tangential_eigen_np = lens_calc.tangential_eigen_value_from(grid=grid_uniform)

assert isinstance(
    tangential_eigen_np, aa.Array2D
), f"tangential_eigen_value_from (numpy): expected aa.Array2D, got {type(tangential_eigen_np)}"

tangential_eigen_jax_fn = jax.jit(
    lambda: lens_calc.tangential_eigen_value_from(grid=grid_uniform, xp=jnp)
)
tangential_eigen_jax = tangential_eigen_jax_fn()

assert isinstance(
    tangential_eigen_jax, jax.Array
), f"tangential_eigen_value_from (jax): expected jax.Array, got {type(tangential_eigen_jax)}"

npt.assert_allclose(
    np.array(tangential_eigen_jax),
    np.array(tangential_eigen_np),
    rtol=5e-3,
    err_msg="tangential_eigen_value_from: numpy vs jax mismatch",
)

"""
__radial_eigen_value_from__

The radial eigenvalue is `1 - convergence + |shear|`.  Return-type behaviour is
identical to `tangential_eigen_value_from`: `aa.Array2D` for the NumPy path,
`jax.Array` for the JAX path inside `jax.jit`.
"""
radial_eigen_np = lens_calc.radial_eigen_value_from(grid=grid_uniform)

assert isinstance(
    radial_eigen_np, aa.Array2D
), f"radial_eigen_value_from (numpy): expected aa.Array2D, got {type(radial_eigen_np)}"

radial_eigen_jax_fn = jax.jit(
    lambda: lens_calc.radial_eigen_value_from(grid=grid_uniform, xp=jnp)
)
radial_eigen_jax = radial_eigen_jax_fn()

assert isinstance(
    radial_eigen_jax, jax.Array
), f"radial_eigen_value_from (jax): expected jax.Array, got {type(radial_eigen_jax)}"

npt.assert_allclose(
    np.array(radial_eigen_jax),
    np.array(radial_eigen_np),
    atol=2e-2,
    err_msg="radial_eigen_value_from: numpy vs jax mismatch",
)

"""
__magnification_2d_from__

`magnification_2d_from` is the uniform-grid counterpart of
`magnification_2d_via_hessian_from`: it wraps the result in `aa.Array2D` (not
`aa.ArrayIrregular`) on the NumPy path, so it also requires a uniform `Grid2D`.

  - `xp=np`  → returns `aa.Array2D`
  - `xp=jnp` → returns a raw `jax.Array` (the `@to_array` decorator skips wrapping)
"""
mag2d_np = lens_calc.magnification_2d_from(grid=grid_uniform)

assert isinstance(
    mag2d_np, aa.Array2D
), f"magnification_2d_from (numpy): expected aa.Array2D, got {type(mag2d_np)}"

mag2d_jax_fn = jax.jit(
    lambda: lens_calc.magnification_2d_from(grid=grid_uniform, xp=jnp)
)
mag2d_jax = mag2d_jax_fn()

assert isinstance(
    mag2d_jax, jax.Array
), f"magnification_2d_from (jax): expected jax.Array, got {type(mag2d_jax)}"

npt.assert_allclose(
    np.array(mag2d_jax),
    np.array(mag2d_np),
    rtol=2e-2,
    err_msg="magnification_2d_from: numpy vs jax mismatch",
)
