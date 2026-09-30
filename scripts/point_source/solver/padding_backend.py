"""
PointSolver Backend and Padding
==============================

Regression for PyAutoLens#759 (cluster arc phase 1b). An omitted padding option
must follow the effective call-time backend. Run from the workspace root; no
external datasets or model fit are needed.

__Contents__
Fixture; unregistered and registered tracer paths; explicit padding choices.

__Env__
ENV: jax full_datasets
"""

import os
import numpy as np
import jax
import jax.numpy as jnp
from scipy.optimize import linear_sum_assignment

import autolens as al
from autolens.jax import register_tracer_classes
from autolens.point.solver.implicit_diff import tracer_is_jax_compatible
from autoarray.structures.triangles.array import MAX_CONTAINING_SIZE

assert (
    os.environ.get("PYAUTO_SMALL_DATASETS") != "1"
), "This regression requires real solves"
source = (0.1, 0.02)
tracer = al.Tracer(
    galaxies=[
        al.Galaxy(redshift=0.5, mass=al.mp.IsothermalSph(einstein_radius=1.0)),
        al.Galaxy(redshift=1.0, point_0=al.ps.Point(centre=source)),
    ]
)


def solver_for(use_jax):
    return al.PointSolver.for_limits_and_scale(
        y_min=-2.0,
        y_max=2.0,
        x_min=-2.0,
        x_max=2.0,
        scale=0.2,
        pixel_scale_precision=0.01,
        use_jax=use_jax,
    )


def finite(array):
    array = np.asarray(array)
    return array[np.isfinite(array).all(axis=1)]


def assert_same_images(array, reference):
    images = finite(array)
    assert len(images) == len(reference) == 2
    distance = np.linalg.norm(images[:, None, :] - reference[None, :, :], axis=-1)
    rows, cols = linear_sum_assignment(distance)
    assert np.max(distance[rows, cols]) < 0.02


reference = finite(solver_for(False).solve(tracer, source, xp=np).array)

"""__Tracer Paths__

The hand-built tracer uses the direct forward path. Registering its classes
then exercises the production custom-JVP path with the same backend matrix.
Padded outputs must retain their static shape under JIT; explicit stripping is
still supported eagerly and is still rejected inside JIT.
"""

for registered in (False, True):
    if registered:
        register_tracer_classes(tracer)
    assert tracer_is_jax_compatible(tracer) is registered
    for constructor_jax in (False, True):
        solver = solver_for(constructor_jax)
        for override in ("numpy", "jax", "omitted"):
            keywords = (
                {}
                if override == "omitted"
                else {"xp": np if override == "numpy" else jnp}
            )
            effective_jax = (
                constructor_jax if override == "omitted" else override == "jax"
            )

            def solve(beta, remove_infinities=None):
                return solver.solve(
                    tracer, beta, remove_infinities=remove_infinities, **keywords
                ).array

            eager = np.asarray(solve(source))
            assert_same_images(eager, reference)
            if effective_jax:
                assert eager.shape == (MAX_CONTAINING_SIZE, 2)
                compiled = np.asarray(jax.jit(solve)(jnp.asarray(source)))
                assert compiled.shape == eager.shape
                np.testing.assert_allclose(compiled, eager, rtol=0, atol=1e-12)
                explicit_padded = np.asarray(solve(source, remove_infinities=False))
                np.testing.assert_allclose(eager, explicit_padded, rtol=0, atol=1e-12)
                stripped = np.asarray(solve(source, remove_infinities=True))
                assert stripped.shape == (2, 2)
                assert_same_images(stripped, reference)
                try:
                    jax.jit(lambda beta: solve(beta, remove_infinities=True))(
                        jnp.asarray(source)
                    )
                except jax.errors.NonConcreteBooleanIndexError:
                    pass
                else:
                    raise AssertionError(
                        "Explicit dynamic stripping unexpectedly succeeded inside JIT"
                    )
            else:
                assert eager.shape == (2, 2)
                # Reject every image so explicit False can be distinguished from
                # the omitted/True stripped result, including a JAX constructor.
                solver.magnification_threshold = 1e100
                assert np.asarray(solve(source)).shape == (0, 2)
                padded = np.asarray(solve(source, remove_infinities=False))
                assert len(padded) > 0 and np.isinf(padded).all()
                assert np.asarray(solve(source, remove_infinities=True)).shape == (0, 2)
                solver.magnification_threshold = 0.1
            print(
                f"PASS registered={registered}, constructor_jax={constructor_jax}, override={override}",
                flush=True,
            )
        jax.clear_caches()
print("All 12 constructor/backend/tracer cases passed")
