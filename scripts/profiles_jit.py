"""
Profiles: JAX JIT
=================

This script tests that light and mass profile methods produce identical results on the
NumPy path and the JAX path when the JAX path is compiled with `jax.jit`.

It is a companion to `hessian_jax.py`, which tests the same pattern for the
`LensCalc` hessian-derived quantities.  This script focuses on the lower-level profile
methods (`image_2d_from`, `deflections_yx_2d_from`, `convergence_2d_from`) that are
called internally by `LensCalc` and by the `Tracer`.

__What is tested__

For every profile and method the following three checks are made, on both a
`Grid2DIrregular` (irregular grid) and a `Grid2D.uniform` (uniform grid):

1. NumPy path: method called with default `xp=np`.  Result is an autoarray type
   (`aa.ArrayIrregular`, `aa.VectorYX2DIrregular`, `aa.Array2D`, `aa.VectorYX2D`)
   whose `._array` attribute is a plain `numpy.ndarray`.

2. JAX path (outside JIT): method called with `xp=jnp`.  Result is the same
   autoarray type, but `._array` is a `jax.Array`.

3. JAX path (inside `jax.jit`): a lambda wraps the method call and extracts `._array`
   before returning, so the JIT output is a raw `jax.Array` (autoarray types are not
   valid JIT outputs).  The result must match the NumPy path to `rtol=1e-5`.

__Profiles covered__

Light:
  - ag.lp.Sersic          → image_2d_from
  - ag.lp.Exponential     → image_2d_from
  - ag.lp.Gaussian        → image_2d_from
  - ag.lp.DevVaucouleurs  → image_2d_from

Linear light profiles (lp_linear):
  - ag.lp_linear.Sersic         → image_2d_from
  - ag.lp_linear.Exponential    → image_2d_from
  - ag.lp_linear.Gaussian       → image_2d_from
  - ag.lp_linear.DevVaucouleurs → image_2d_from

Mass:
  - ag.mp.Isothermal      → deflections_yx_2d_from, convergence_2d_from
  - ag.mp.PowerLaw        → deflections_yx_2d_from, convergence_2d_from
  - ag.mp.NFW             → deflections_yx_2d_from, convergence_2d_from
  - ag.mp.ExternalShear   → deflections_yx_2d_from, convergence_2d_from
  - ag.mp.ExternalPotential → deflections_yx_2d_from, convergence_2d_from
"""

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import autoarray as aa
import autogalaxy as ag

"""
__Grids__

Two grids are used throughout:

- `grid_irr`:    a `Grid2DIrregular` with a handful of off-centre points.  The points
  are chosen to avoid profile centres so that there are no singularities or NaNs.

- `grid_uni`:    a small `Grid2D.uniform` grid, shifted off-centre for the same reason.
  A shift of (0.5, 0.5) arcsec ensures no grid point falls exactly on any profile
  centre at (0.0, 0.0).
"""
grid_irr = ag.Grid2DIrregular(values=[(0.5, 0.5), (1.0, -0.5), (-0.5, 1.0), (1.5, 1.5)])
grid_uni = ag.Grid2D.uniform(shape_native=(5, 5), pixel_scales=0.3)

"""
__Helper__

`check_profile_method` factors out the three-step check described in the module
docstring so that each profile + method combination is a single call.

The `array_attr` argument selects which attribute of the autoarray result holds the
raw underlying data (always `._array` for the types we encounter here).
"""


def check_profile_method(
    label: str,
    profile,
    method_name: str,
    grid,
    np_type,
    jax_type,
    rtol: float = 1e-5,
    atol: float = 0.0,
):
    """
    Run the three-step JAX-JIT check for one profile method on one grid.

    Parameters
    ----------
    label
        Human-readable identifier used in assertion error messages.
    profile
        The instantiated profile object.
    method_name
        Name of the method to call (e.g. "image_2d_from").
    grid
        The grid passed to the method.
    np_type
        Expected autoarray type on the NumPy path.
    jax_type
        Expected autoarray type on the JAX path (outside JIT).
    rtol
        Relative tolerance for the NumPy vs JAX numerical comparison.
    atol
        Absolute tolerance for the NumPy vs JAX numerical comparison. Needed
        for profiles whose output legitimately crosses zero on a grid point
        (e.g. ``mp.ExternalPotential`` convergence on the τ-null line), where
        the rtol-only check blows up on sub-machine-precision reductions.
    """
    method = getattr(profile, method_name)

    # --- 1. NumPy path ---
    result_np = method(grid=grid)
    assert isinstance(
        result_np, np_type
    ), f"{label} (numpy): expected {np_type.__name__}, got {type(result_np)}"
    assert isinstance(
        result_np._array, np.ndarray
    ), f"{label} (numpy): ._array expected np.ndarray, got {type(result_np._array)}"

    # --- 2. JAX path outside JIT ---
    result_jax_outer = method(grid=grid, xp=jnp)
    assert isinstance(
        result_jax_outer, jax_type
    ), f"{label} (jax outer): expected {jax_type.__name__}, got {type(result_jax_outer)}"
    assert isinstance(
        result_jax_outer._array, jax.Array
    ), f"{label} (jax outer): ._array expected jax.Array, got {type(result_jax_outer._array)}"

    # --- 3. JAX path inside jax.jit ---
    # Extract ._array at the JIT boundary so the output is a raw jax.Array.
    # Autoarray types are not valid JAX pytree outputs and cannot be returned
    # directly from a jax.jit-compiled function.
    jitted_fn = jax.jit(lambda: method(grid=grid, xp=jnp)._array)
    result_jax_jit = jitted_fn()

    assert isinstance(
        result_jax_jit, jax.Array
    ), f"{label} (jax jit): expected jax.Array, got {type(result_jax_jit)}"

    npt.assert_allclose(
        np.array(result_jax_jit),
        np.array(result_np._array),
        rtol=rtol,
        atol=atol,
        err_msg=f"{label}: numpy vs jax (jit) mismatch",
    )


"""
__Light Profiles__

Each light profile is tested with `image_2d_from` on both the irregular and uniform
grids.  The expected autoarray types are:

  - `Grid2DIrregular` input → `aa.ArrayIrregular`
  - `Grid2D` input          → `aa.Array2D`
"""

print("Testing light profiles...")

"""
ag.lp.Sersic
"""
sersic = ag.lp.Sersic(
    centre=(0.0, 0.0),
    ell_comps=(0.0, 0.111111),
    intensity=0.1,
    effective_radius=0.6,
    sersic_index=2.5,
)

check_profile_method(
    label="lp.Sersic.image_2d_from (irregular)",
    profile=sersic,
    method_name="image_2d_from",
    grid=grid_irr,
    np_type=aa.ArrayIrregular,
    jax_type=aa.ArrayIrregular,
)
check_profile_method(
    label="lp.Sersic.image_2d_from (uniform)",
    profile=sersic,
    method_name="image_2d_from",
    grid=grid_uni,
    np_type=aa.Array2D,
    jax_type=aa.Array2D,
)

print("  lp.Sersic OK")

"""
ag.lp.Exponential
"""
exponential = ag.lp.Exponential(
    centre=(0.0, 0.0),
    ell_comps=(0.0, 0.0),
    intensity=0.1,
    effective_radius=0.8,
)

check_profile_method(
    label="lp.Exponential.image_2d_from (irregular)",
    profile=exponential,
    method_name="image_2d_from",
    grid=grid_irr,
    np_type=aa.ArrayIrregular,
    jax_type=aa.ArrayIrregular,
)
check_profile_method(
    label="lp.Exponential.image_2d_from (uniform)",
    profile=exponential,
    method_name="image_2d_from",
    grid=grid_uni,
    np_type=aa.Array2D,
    jax_type=aa.Array2D,
)

print("  lp.Exponential OK")

"""
ag.lp.Gaussian
"""
gaussian = ag.lp.Gaussian(
    centre=(0.0, 0.0),
    ell_comps=(0.0, 0.0),
    intensity=0.5,
    sigma=0.5,
)

check_profile_method(
    label="lp.Gaussian.image_2d_from (irregular)",
    profile=gaussian,
    method_name="image_2d_from",
    grid=grid_irr,
    np_type=aa.ArrayIrregular,
    jax_type=aa.ArrayIrregular,
)
check_profile_method(
    label="lp.Gaussian.image_2d_from (uniform)",
    profile=gaussian,
    method_name="image_2d_from",
    grid=grid_uni,
    np_type=aa.Array2D,
    jax_type=aa.Array2D,
)

print("  lp.Gaussian OK")

"""
ag.lp.DevVaucouleurs
"""
dev_vaucouleurs = ag.lp.DevVaucouleurs(
    centre=(0.0, 0.0),
    ell_comps=(0.0, 0.0),
    intensity=0.1,
    effective_radius=0.4,
)

check_profile_method(
    label="lp.DevVaucouleurs.image_2d_from (irregular)",
    profile=dev_vaucouleurs,
    method_name="image_2d_from",
    grid=grid_irr,
    np_type=aa.ArrayIrregular,
    jax_type=aa.ArrayIrregular,
)
check_profile_method(
    label="lp.DevVaucouleurs.image_2d_from (uniform)",
    profile=dev_vaucouleurs,
    method_name="image_2d_from",
    grid=grid_uni,
    np_type=aa.Array2D,
    jax_type=aa.Array2D,
)

print("  lp.DevVaucouleurs OK")

"""
__Linear Light Profiles__

Linear light profiles (`lp_linear.*`) are identical to their `lp.*` counterparts
except that `intensity` is absent — it is solved by linear inversion during
model fitting. The JAX JIT path must work identically.
"""

print("Testing linear light profiles...")

"""
ag.lp_linear.Sersic
"""
lp_linear_sersic = ag.lp_linear.Sersic(
    centre=(0.0, 0.0),
    ell_comps=(0.0, 0.111111),
    effective_radius=0.6,
    sersic_index=2.5,
)

check_profile_method(
    label="lp_linear.Sersic.image_2d_from (irregular)",
    profile=lp_linear_sersic,
    method_name="image_2d_from",
    grid=grid_irr,
    np_type=aa.ArrayIrregular,
    jax_type=aa.ArrayIrregular,
)
check_profile_method(
    label="lp_linear.Sersic.image_2d_from (uniform)",
    profile=lp_linear_sersic,
    method_name="image_2d_from",
    grid=grid_uni,
    np_type=aa.Array2D,
    jax_type=aa.Array2D,
)

print("  lp_linear.Sersic OK")

"""
ag.lp_linear.Exponential
"""
lp_linear_exponential = ag.lp_linear.Exponential(
    centre=(0.0, 0.0),
    ell_comps=(0.0, 0.0),
    effective_radius=0.8,
)

check_profile_method(
    label="lp_linear.Exponential.image_2d_from (irregular)",
    profile=lp_linear_exponential,
    method_name="image_2d_from",
    grid=grid_irr,
    np_type=aa.ArrayIrregular,
    jax_type=aa.ArrayIrregular,
)
check_profile_method(
    label="lp_linear.Exponential.image_2d_from (uniform)",
    profile=lp_linear_exponential,
    method_name="image_2d_from",
    grid=grid_uni,
    np_type=aa.Array2D,
    jax_type=aa.Array2D,
)

print("  lp_linear.Exponential OK")

"""
ag.lp_linear.Gaussian
"""
lp_linear_gaussian = ag.lp_linear.Gaussian(
    centre=(0.0, 0.0),
    ell_comps=(0.0, 0.0),
    sigma=0.5,
)

check_profile_method(
    label="lp_linear.Gaussian.image_2d_from (irregular)",
    profile=lp_linear_gaussian,
    method_name="image_2d_from",
    grid=grid_irr,
    np_type=aa.ArrayIrregular,
    jax_type=aa.ArrayIrregular,
)
check_profile_method(
    label="lp_linear.Gaussian.image_2d_from (uniform)",
    profile=lp_linear_gaussian,
    method_name="image_2d_from",
    grid=grid_uni,
    np_type=aa.Array2D,
    jax_type=aa.Array2D,
)

print("  lp_linear.Gaussian OK")

"""
ag.lp_linear.DevVaucouleurs
"""
lp_linear_dev_vaucouleurs = ag.lp_linear.DevVaucouleurs(
    centre=(0.0, 0.0),
    ell_comps=(0.0, 0.0),
    effective_radius=0.4,
)

check_profile_method(
    label="lp_linear.DevVaucouleurs.image_2d_from (irregular)",
    profile=lp_linear_dev_vaucouleurs,
    method_name="image_2d_from",
    grid=grid_irr,
    np_type=aa.ArrayIrregular,
    jax_type=aa.ArrayIrregular,
)
check_profile_method(
    label="lp_linear.DevVaucouleurs.image_2d_from (uniform)",
    profile=lp_linear_dev_vaucouleurs,
    method_name="image_2d_from",
    grid=grid_uni,
    np_type=aa.Array2D,
    jax_type=aa.Array2D,
)

print("  lp_linear.DevVaucouleurs OK")

"""
__Mass Profiles__

Each mass profile is tested for both `deflections_yx_2d_from` and `convergence_2d_from`.

Expected autoarray types:

  deflections_yx_2d_from:
    - `Grid2DIrregular` → `aa.VectorYX2DIrregular`
    - `Grid2D`          → `aa.VectorYX2D`

  convergence_2d_from:
    - `Grid2DIrregular` → `aa.ArrayIrregular`
    - `Grid2D`          → `aa.Array2D`

The NFW profile uses an analytic JAX implementation so a looser tolerance
`rtol=1e-4` is appropriate; all others use `rtol=1e-5`.
"""

print("Testing mass profiles...")

"""
ag.mp.Isothermal
"""
isothermal = ag.mp.Isothermal(
    centre=(0.0, 0.0),
    ell_comps=(0.0, 0.111111),
    einstein_radius=1.5,
)

for method_name, np_irr, jax_irr, np_uni, jax_uni in [
    (
        "deflections_yx_2d_from",
        aa.VectorYX2DIrregular,
        aa.VectorYX2DIrregular,
        aa.VectorYX2D,
        aa.VectorYX2D,
    ),
    (
        "convergence_2d_from",
        aa.ArrayIrregular,
        aa.ArrayIrregular,
        aa.Array2D,
        aa.Array2D,
    ),
]:
    check_profile_method(
        label=f"mp.Isothermal.{method_name} (irregular)",
        profile=isothermal,
        method_name=method_name,
        grid=grid_irr,
        np_type=np_irr,
        jax_type=jax_irr,
    )
    check_profile_method(
        label=f"mp.Isothermal.{method_name} (uniform)",
        profile=isothermal,
        method_name=method_name,
        grid=grid_uni,
        np_type=np_uni,
        jax_type=jax_uni,
    )

print("  mp.Isothermal OK")

"""
ag.mp.PowerLaw
"""
power_law = ag.mp.PowerLaw(
    centre=(0.0, 0.0),
    ell_comps=(0.0, 0.111111),
    einstein_radius=1.5,
    slope=2.2,
)

for method_name, np_irr, jax_irr, np_uni, jax_uni in [
    (
        "deflections_yx_2d_from",
        aa.VectorYX2DIrregular,
        aa.VectorYX2DIrregular,
        aa.VectorYX2D,
        aa.VectorYX2D,
    ),
    (
        "convergence_2d_from",
        aa.ArrayIrregular,
        aa.ArrayIrregular,
        aa.Array2D,
        aa.Array2D,
    ),
]:
    check_profile_method(
        label=f"mp.PowerLaw.{method_name} (irregular)",
        profile=power_law,
        method_name=method_name,
        grid=grid_irr,
        np_type=np_irr,
        jax_type=jax_irr,
    )
    check_profile_method(
        label=f"mp.PowerLaw.{method_name} (uniform)",
        profile=power_law,
        method_name=method_name,
        grid=grid_uni,
        np_type=np_uni,
        jax_type=jax_uni,
    )

print("  mp.PowerLaw OK")

"""
ag.mp.NFW

The NFW deflections use an analytic JAX implementation (`deflections_2d_via_analytic_from`)
so the tolerance is relaxed to `rtol=1e-4`.
"""
nfw = ag.mp.NFW(
    centre=(0.0, 0.0),
    ell_comps=(0.0, 0.0),
    kappa_s=0.05,
    scale_radius=1.0,
)

for method_name, np_irr, jax_irr, np_uni, jax_uni, rtol in [
    (
        "deflections_yx_2d_from",
        aa.VectorYX2DIrregular,
        aa.VectorYX2DIrregular,
        aa.VectorYX2D,
        aa.VectorYX2D,
        1e-4,
    ),
    (
        "convergence_2d_from",
        aa.ArrayIrregular,
        aa.ArrayIrregular,
        aa.Array2D,
        aa.Array2D,
        1e-4,
    ),
]:
    check_profile_method(
        label=f"mp.NFW.{method_name} (irregular)",
        profile=nfw,
        method_name=method_name,
        grid=grid_irr,
        np_type=np_irr,
        jax_type=jax_irr,
        rtol=rtol,
    )
    check_profile_method(
        label=f"mp.NFW.{method_name} (uniform)",
        profile=nfw,
        method_name=method_name,
        grid=grid_uni,
        np_type=np_uni,
        jax_type=jax_uni,
        rtol=rtol,
    )

print("  mp.NFW OK")

"""
ag.mp.ExternalShear
"""
external_shear = ag.mp.ExternalShear(gamma_1=0.05, gamma_2=0.03)

for method_name, np_irr, jax_irr, np_uni, jax_uni in [
    (
        "deflections_yx_2d_from",
        aa.VectorYX2DIrregular,
        aa.VectorYX2DIrregular,
        aa.VectorYX2D,
        aa.VectorYX2D,
    ),
    (
        "convergence_2d_from",
        aa.ArrayIrregular,
        aa.ArrayIrregular,
        aa.Array2D,
        aa.Array2D,
    ),
]:
    check_profile_method(
        label=f"mp.ExternalShear.{method_name} (irregular)",
        profile=external_shear,
        method_name=method_name,
        grid=grid_irr,
        np_type=np_irr,
        jax_type=jax_irr,
    )
    check_profile_method(
        label=f"mp.ExternalShear.{method_name} (uniform)",
        profile=external_shear,
        method_name=method_name,
        grid=grid_uni,
        np_type=np_uni,
        jax_type=jax_uni,
    )

print("  mp.ExternalShear OK")

"""
ag.mp.ExternalPotential
"""
external_potential = ag.mp.ExternalPotential(
    centre=(0.1, 0.2),
    gamma_1=0.05,
    gamma_2=0.03,
    tau_1=0.02,
    tau_2=-0.01,
    delta_1=0.015,
    delta_2=0.01,
)

for method_name, np_irr, jax_irr, np_uni, jax_uni in [
    (
        "deflections_yx_2d_from",
        aa.VectorYX2DIrregular,
        aa.VectorYX2DIrregular,
        aa.VectorYX2D,
        aa.VectorYX2D,
    ),
    (
        "convergence_2d_from",
        aa.ArrayIrregular,
        aa.ArrayIrregular,
        aa.Array2D,
        aa.Array2D,
    ),
]:
    check_profile_method(
        label=f"mp.ExternalPotential.{method_name} (irregular)",
        profile=external_potential,
        method_name=method_name,
        grid=grid_irr,
        np_type=np_irr,
        jax_type=jax_irr,
        atol=1e-12,
    )
    check_profile_method(
        label=f"mp.ExternalPotential.{method_name} (uniform)",
        profile=external_potential,
        method_name=method_name,
        grid=grid_uni,
        np_type=np_uni,
        jax_type=jax_uni,
        atol=1e-12,
    )

print("  mp.ExternalPotential OK")

print("\nAll profiles_jit.py checks passed.")
