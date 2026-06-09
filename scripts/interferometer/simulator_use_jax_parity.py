"""
SimulatorInterferometer use_jax parity test
============================================

Runs ``al.SimulatorInterferometer(use_jax=False)`` and ``al.SimulatorInterferometer(use_jax=True)``
against the same tracer + grid and asserts the noise-free simulated visibilities
agree to machine precision. Cross-xp numerical validation for
``SimulatorInterferometer.use_jax=True`` — library unit tests stay NumPy-only.

Noise is disabled (``noise_sigma=None``) because JAX and NumPy use different RNG
algorithms; the same seed produces different draws. This test covers the
deterministic Fourier-transform path.

Also tests the @jax.jit roundtrip for the interferometer simulator.
"""

from autoconf import jax_wrapper  # Sets JAX float64 before other imports

import jax
import numpy as np

import autolens as al
from autolens.jax import register_tracer_classes


grid = al.Grid2D.uniform(shape_native=(64, 64), pixel_scales=0.1)

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

uv_wavelengths = np.random.RandomState(seed=0).uniform(
    low=-1000.0, high=1000.0, size=(200, 2)
)

# Noise-free configuration: both backends produce identical deterministic visibilities.
common_kwargs = dict(
    uv_wavelengths=uv_wavelengths,
    exposure_time=300.0,
    noise_sigma=None,  # disable noise for deterministic parity
)

# NumPy path.
simulator_np = al.SimulatorInterferometer(**common_kwargs)
dataset_np = simulator_np.via_tracer_from(tracer=tracer, grid=grid)
data_np = np.asarray(dataset_np.data)

# JAX path (eager).
simulator_jax = al.SimulatorInterferometer(use_jax=True, **common_kwargs)
dataset_jax = simulator_jax.via_tracer_from(tracer=tracer, grid=grid)
data_jax = np.asarray(dataset_jax.data)

assert (
    data_np.shape == data_jax.shape
), f"Shape mismatch: numpy={data_np.shape} vs jax={data_jax.shape}"
np.testing.assert_allclose(
    data_np,
    data_jax,
    atol=1e-8,
    err_msg="SimulatorInterferometer(use_jax=True) visibilities differ from use_jax=False",
)

print(
    f"PASS: SimulatorInterferometer(use_jax=True) visibilities match use_jax=False "
    f"to atol=1e-8 ({data_np.shape[0]} visibilities)."
)

# @jax.jit roundtrip.
register_tracer_classes(tracer)


@jax.jit
def simulate_jit(tracer):
    dataset = simulator_jax.via_tracer_from(tracer=tracer, grid=grid)
    return dataset.data._array


data_jit = np.asarray(simulate_jit(tracer))

np.testing.assert_allclose(
    data_jax,
    data_jit,
    atol=1e-8,
    err_msg="SimulatorInterferometer @jax.jit visibilities differ from eager JAX path",
)
print(
    f"PASS: SimulatorInterferometer @jax.jit roundtrip matches eager JAX "
    f"to atol=1e-8 ({data_jit.shape[0]} visibilities)."
)
