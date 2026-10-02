"""Finite forward values and gradients for discarded fit-utility divisions."""

import jax
import jax.numpy as jnp
import numpy as np

from autoarray.fit import fit_util

jax.config.update("jax_enable_x64", True)


def check(name, function, denominator, included, squared=False):
    residual = np.array([2.0, -3.0, 4.0, 5.0])
    safe = np.where(included, denominator, 1.0)
    expected = np.where(included, residual / safe, 0.0)
    numerator_grad = np.where(included, 1.0 / safe, 0.0)
    denominator_grad = np.where(included, -residual / safe**2, 0.0)
    if squared:
        expected = expected**2
        numerator_grad = np.where(included, 2.0 * residual / safe**2, 0.0)
        denominator_grad = np.where(included, -2.0 * residual**2 / safe**3, 0.0)
    np.testing.assert_allclose(function(residual, denominator, np), expected)
    evaluate = lambda r, d: function(r, d, jnp)
    gradient = jax.grad(lambda r, d: jnp.sum(evaluate(r, d)), argnums=(0, 1))
    for mode, forward, grad in (
        ("eager", evaluate, gradient),
        ("jit", jax.jit(evaluate), jax.jit(gradient)),
    ):
        np.testing.assert_allclose(
            forward(jnp.asarray(residual), jnp.asarray(denominator)), expected
        )
        for actual, analytic in zip(
            grad(jnp.asarray(residual), jnp.asarray(denominator)),
            (numerator_grad, denominator_grad),
        ):
            assert np.isfinite(
                actual
            ).all(), f"{name} {mode}: non-finite gradient {actual}"
            np.testing.assert_allclose(actual, analytic)
            np.testing.assert_array_equal(np.asarray(actual)[~included], 0.0)
    print(f"PASS: {name} NumPy, eager/JIT forwards and numerator/denominator gradients")


mask = np.array([False, False, True, True])
cases = (
    (
        "chi_squared",
        lambda r, d, xp: fit_util.chi_squared_map_with_mask_from(
            residual_map=r, noise_map=d, mask=mask, xp=xp
        ),
        np.array([1.0, 2.0, 0.0, 0.0]),
        ~mask,
        True,
    ),
    (
        "residual_fraction",
        lambda r, d, xp: fit_util.residual_flux_fraction_map_from(
            residual_map=r, data=d, xp=xp
        ),
        np.array([1.0, 0.0, -2.0, 0.0]),
        np.array([True, False, True, False]),
        False,
    ),
    (
        "masked_residual_fraction",
        lambda r, d, xp: fit_util.residual_flux_fraction_map_with_mask_from(
            residual_map=r, data=d, mask=mask, xp=xp
        ),
        np.array([1.0, 0.0, -2.0, 0.0]),
        np.array([True, False, False, False]),
        False,
    ),
)
failures = []
for case in cases:
    try:
        check(*case)
    except AssertionError as error:
        failures.append(f"{case[0]}: {error}")
assert not failures, "\n".join(failures)
