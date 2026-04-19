"""
Path A PoC: jit-wrap ``analysis.fit_from`` for MGE (Interferometer)
====================================================================

Sibling of ``imaging/mge_pytree.py`` for the interferometer / uv-plane path
(see admin_jammy/prompt/issued/fit_interferometer_pytree_mge.md).

Swaps ``Imaging`` → ``Interferometer`` and ``AnalysisImaging`` → ``AnalysisInterferometer``
(with ``TransformerDFT``). Same MGE source and simple Isothermal + ExternalShear lens
as ``interferometer/mge.py``.

Success criterion:
  - ``jax.jit(analysis.fit_from)(instance)`` returns a ``FitInterferometer`` whose
    ``log_likelihood`` is a ``jax.Array`` matching the NumPy-path scalar.
"""

from os import path

import jax
import jax.numpy as jnp
import numpy as np

import autofit as af
import autolens as al

from autofit.jax.pytrees import enable_pytrees, register_model

enable_pytrees()


"""
__Real-space mask__
"""
mask_radius = 3.0

real_space_mask = al.Mask2D.circular(
    shape_native=(256, 256),
    pixel_scales=0.1,
    radius=mask_radius,
)


"""
__Dataset__ — same on-disk dataset used by ``interferometer/mge.py``.
"""
dataset_name = "simple"
dataset_path = path.join("dataset", "interferometer", dataset_name)

if al.util.dataset.should_simulate(dataset_path):
    import subprocess
    import sys

    subprocess.run(
        [
            sys.executable,
            "scripts/jax_likelihood_functions/interferometer/simulator.py",
        ],
        check=True,
    )

dataset = al.Interferometer.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    uv_wavelengths_path=path.join(dataset_path, "uv_wavelengths.fits"),
    real_space_mask=real_space_mask,
    transformer_class=al.TransformerDFT,
)


"""
__Model__ — Isothermal + ExternalShear lens, MGE source.
"""
mass = af.Model(al.mp.Isothermal)
shear = af.Model(al.mp.ExternalShear)

lens = af.Model(al.Galaxy, redshift=0.5, mass=mass, shear=shear)

bulge = al.model_util.mge_model_from(
    mask_radius=mask_radius, total_gaussians=10, centre_prior_is_uniform=False
)

source = af.Model(al.Galaxy, redshift=1.0, bulge=bulge)

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))


"""
__Analysis__ on the JAX path (``use_jax=True``).
"""
analysis = al.AnalysisInterferometer(
    dataset=dataset,
    use_jax=True,
)

register_model(model)

instance = model.instance_from_prior_medians()


"""
__NumPy reference scalar__.
"""
analysis_np = al.AnalysisInterferometer(
    dataset=dataset,
    use_jax=False,
)
fit_np = analysis_np.fit_from(instance=instance)
print("NumPy fit.log_likelihood:", float(fit_np.log_likelihood))


"""
__Path A: jit-wrap ``analysis.fit_from``__.
"""
fit_jit_fn = jax.jit(analysis.fit_from)
fit = fit_jit_fn(instance)

print("JIT fit type:", type(fit).__name__)
print("JIT fit.log_likelihood:", fit.log_likelihood)
assert isinstance(fit.log_likelihood, jnp.ndarray), (
    f"expected jax.Array, got {type(fit.log_likelihood)}"
)
np.testing.assert_allclose(
    float(fit.log_likelihood), float(fit_np.log_likelihood), rtol=1e-4
)

print("PASS: jit(fit_from) round-trip matches NumPy scalar.")
