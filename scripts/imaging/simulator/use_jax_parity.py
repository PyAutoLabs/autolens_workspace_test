"""
SimulatorImaging use_jax parity test
=====================================

Runs ``al.SimulatorImaging(use_jax=False)`` and ``al.SimulatorImaging(use_jax=True)``
against the same tracer + grid and asserts the noise-free simulated image agrees
to machine precision. Cross-xp numerical validation for ``SimulatorImaging.use_jax=True``
— library unit tests stay NumPy-only.

Noise is disabled on both paths because JAX and NumPy use different RNG algorithms;
the same seed produces different draws. The simulator's xp-threaded noise path is
tested separately (``test_autoarray/dataset/imaging/test_simulator_use_jax.py``
covers the constructor wiring; this script covers numerical agreement of the
deterministic image-generation path).

This script intentionally stops at eager simulator parity. Lower-level tracer
and lens-calc scripts cover ``jax.jit`` directly; the imaging simulator still
rebuilds native ``Array2D`` wrappers with NumPy indexing, which is not a valid
JIT boundary for traced image values.
"""

from autolens import jax_wrapper  # Sets JAX float64 before other imports

import numpy as np

import autolens as al


grid = al.Grid2D.uniform(shape_native=(100, 100), pixel_scales=0.1)

lens_galaxy = al.Galaxy(
    redshift=0.5,
    light=al.lp.Sersic(
        centre=(0.0, 0.0),
        ell_comps=(0.01, 0.01),
        intensity=1.0,
        effective_radius=0.5,
        sersic_index=4.0,
    ),
    mass=al.mp.Isothermal(
        centre=(0.0, 0.0),
        ell_comps=(0.01, 0.01),
        einstein_radius=1.6,
    ),
)
source_galaxy = al.Galaxy(
    redshift=1.0,
    light=al.lp.Sersic(
        centre=(0.1, 0.1),
        ell_comps=(0.01, 0.01),
        intensity=0.5,
        effective_radius=0.2,
        sersic_index=1.0,
    ),
)
tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

psf = al.Convolver.from_gaussian(
    shape_native=(11, 11), sigma=0.05, pixel_scales=grid.pixel_scales
)

# Noise-free configuration: both backends must produce identical deterministic images.
common_kwargs = dict(
    exposure_time=300.0,
    psf=psf,
    background_sky_level=0.1,
    add_poisson_noise_to_data=False,
    include_poisson_noise_in_noise_map=False,
)

# NumPy path.
simulator_np = al.SimulatorImaging(**common_kwargs)
dataset_np = simulator_np.via_tracer_from(tracer=tracer, grid=grid)
data_np = np.asarray(dataset_np.data.array)

# JAX path (eager, not yet under @jax.jit).
simulator_jax = al.SimulatorImaging(use_jax=True, **common_kwargs)
dataset_jax = simulator_jax.via_tracer_from(tracer=tracer, grid=grid)
data_jax = np.asarray(dataset_jax.data.array)

assert (
    data_np.shape == data_jax.shape
), f"Shape mismatch: numpy={data_np.shape} vs jax={data_jax.shape}"
np.testing.assert_allclose(
    data_np,
    data_jax,
    atol=1e-8,
    err_msg="SimulatorImaging(use_jax=True) data differs from use_jax=False",
)
print(
    f"PASS: SimulatorImaging(use_jax=True) data matches use_jax=False "
    f"to atol=1e-8 (shape {data_np.shape})."
)
