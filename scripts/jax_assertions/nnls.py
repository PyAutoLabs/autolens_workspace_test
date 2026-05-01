"""
Jax Assertions: NNLS (Positive-Only Solver)
===========================================

Verifies the Jacobi-preconditioned ``jaxnnls`` path through
``aa.util.inversion.reconstruction_positive_only_from`` with ``xp=jnp``:

1. **Ill-conditioned curvature matrix** — Without preconditioning the raw
   ``jaxnnls`` backward pass returns NaN gradients. The Jacobi
   preconditioning re-parameterises the NNLS problem so the solve converges
   and ``jax.value_and_grad`` produces finite gradients.

2. **Well-conditioned curvature matrix** — Jacobi preconditioning is a
   change of coordinates, so the forward primal solution must match the raw
   ``jaxnnls.solve_nnls_primal`` result to within solver tolerance.

Previously: two ``test__reconstruction_positive_only_from__jax_*`` tests in
``test_autoarray/inversion/inversion/test_inversion_util.py``.
"""

import jax
import jax.numpy as jnp
import jaxnnls
import numpy as np
import autoarray as aa

"""
__Ill-conditioned: Finite Gradients via Jacobi Preconditioning__

Build a small symmetric positive-definite Q with cond(Q) ~ 1e7. Without
preconditioning this is enough to break the raw ``jaxnnls`` backward pass.
"""
rng = np.random.default_rng(0)
n = 10
U, _ = np.linalg.qr(rng.standard_normal((n, n)))
eigs = np.logspace(-4, 3, n)
Q_np = (U * eigs) @ U.T
Q_np = 0.5 * (Q_np + Q_np.T)
q_np = rng.standard_normal(n)

Q = jnp.array(Q_np)
q = jnp.array(q_np)


def loss(q_in):
    x = aa.util.inversion.reconstruction_positive_only_from(
        data_vector=q_in,
        curvature_reg_matrix=Q,
        xp=jnp,
    )
    return jnp.sum(x)


value, grad = jax.value_and_grad(loss)(q)

assert np.isfinite(float(value))
grad_np = np.array(grad)
assert np.all(np.isfinite(grad_np)), (
    f"gradient has {np.sum(~np.isfinite(grad_np))} non-finite entries"
)

"""
__Well-conditioned: Preconditioned Solve Matches Raw jaxnnls__

For a moderately-conditioned problem where the raw solver also converges,
the preconditioned solution must match ``jaxnnls.solve_nnls_primal`` to
solver tolerance.
"""
rng = np.random.default_rng(1)
n = 8
U, _ = np.linalg.qr(rng.standard_normal((n, n)))
eigs = np.linspace(0.5, 5.0, n)  # well-conditioned
Q_np = (U * eigs) @ U.T
Q_np = 0.5 * (Q_np + Q_np.T)
q_np = rng.standard_normal(n)

Q = jnp.array(Q_np)
q = jnp.array(q_np)

x_raw = np.array(jaxnnls.solve_nnls_primal(Q, q))
x_pc = np.array(
    aa.util.inversion.reconstruction_positive_only_from(
        data_vector=q,
        curvature_reg_matrix=Q,
        xp=jnp,
    )
)

np.testing.assert_allclose(x_pc, x_raw, rtol=1e-6, atol=1e-8)

print("nnls: all assertions passed")
