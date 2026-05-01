"""
Jax Assertions: AbstractNDArray Pytree Registration
===================================================

Tests gated JAX pytree registration of ``AbstractNDArray`` subclasses,
following the three-step pattern from ``hessian_jax.py``:

1. NumPy path — confirm autoarray type with ``np.ndarray`` backing, no
   pytree registration.
2. JAX path outside JIT — same autoarray type with ``jax.Array`` backing;
   pytree registered.
3. JAX path through ``jax.jit`` — round-trip the instance and assert the
   output carries a ``jax.Array`` leaf.

Previously: ``test_autoarray/test_jax_pytree.py``.
"""

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt

from autoarray.abstract_ndarray import AbstractNDArray, _pytree_registered_classes


class _LeafArray(AbstractNDArray):
    """Minimal concrete ``AbstractNDArray`` with no nested autoarray children.

    Isolates the pytree-registration machinery from the larger autoarray
    hierarchy: a real ``Array2D`` also carries a ``Mask2D`` and other nested
    ``AbstractNDArray`` children whose own registration is covered by
    follow-up steps in the ``fit-imaging-pytree`` task.
    """

    @property
    def native(self):
        return self


"""
__NumPy Path: No Pytree Registration__
"""
_pytree_registered_classes.discard(_LeafArray)

arr = _LeafArray(np.array([1.0, 2.0, 3.0]))

assert isinstance(arr._array, np.ndarray)
assert _LeafArray not in _pytree_registered_classes

"""
__JAX Path Outside JIT: Pytree Registered Once__

Constructing on the JAX path registers the class. A second construction is
a no-op; the class stays registered.
"""
_pytree_registered_classes.discard(_LeafArray)

arr_jax = _LeafArray(jnp.array([1.0, 2.0, 3.0]), xp=jnp)

assert isinstance(arr_jax._array, jnp.ndarray)
assert _LeafArray in _pytree_registered_classes

_LeafArray(jnp.array([4.0, 5.0]), xp=jnp)
assert _LeafArray in _pytree_registered_classes

"""
__JAX Path Through jax.jit: Round-trip Returns Wrapper with jax.Array Leaf__
"""
arr_jax = _LeafArray(jnp.array([1.0, 2.0, 3.0]), xp=jnp)
assert _LeafArray in _pytree_registered_classes

result = jax.jit(lambda a: a)(arr_jax)

assert isinstance(result, _LeafArray)
assert isinstance(result._array, jnp.ndarray)
npt.assert_allclose(np.asarray(result._array), np.asarray(arr_jax._array))

print("pytree: all assertions passed")
