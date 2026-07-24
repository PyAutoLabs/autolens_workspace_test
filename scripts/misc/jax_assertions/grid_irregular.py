"""
Jax Assertions: Grid2DIrregular xp Propagation
==============================================

Verifies that ``Grid2DIrregular.grid_2d_via_deflection_grid_from`` propagates
the array backend (``np`` or ``jnp``) of the receiver through to the result.
This matters because downstream code (e.g. ``.square`` calls in mass profile
deflection arithmetic) needs to keep using the same backend.

The numpy half of this assertion lives in
``test_autoarray/structures/grids/test_irregular_2d.py::test__grid_2d_via_deflection_grid_from__propagates_xp``;
this script holds the JAX half so the unit test stays numpy-only.
"""

import jax.numpy as jnp
import numpy as np
import autoarray as aa

"""
__JAX-Backed Receiver -> JAX-Backed Result__
"""
grid_jax = aa.Grid2DIrregular(values=jnp.array([[1.0, 1.0], [2.0, 2.0]]), xp=jnp)
result_jax = grid_jax.grid_2d_via_deflection_grid_from(
    deflection_grid=jnp.array([[1.0, 0.0], [1.0, 1.0]])
)
assert result_jax._xp is jnp

print("grid_irregular: all assertions passed")
