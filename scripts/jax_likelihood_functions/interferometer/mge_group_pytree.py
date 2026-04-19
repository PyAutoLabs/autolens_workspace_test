"""
Path A PoC: jit-wrap ``analysis.fit_from`` for interferometer + MGE + extra galaxies
=====================================================================================

Intersection of two already-shipped siblings:

- ``interferometer/mge_pytree.py`` — ``FitInterferometer`` + ``TransformerDFT`` +
  MGE source pytree registrations.
- ``imaging/mge_group_pytree.py`` — group-scale ``Tracer`` with extra galaxies.

Model:
  - Lens: Isothermal + ExternalShear (z=0.5)
  - Source: MGE bulge (10 Gaussians) (z=1.0)
  - 3 extra galaxies: IsothermalSph at fixed centres (z=0.5)

Task prompt (``admin_jammy/prompt/issued/fit_interferometer_pytree_mge_group.md``):
if the two siblings landed clean, this should pass with no new registrations.

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
__Dataset__ — same on-disk dataset used by ``interferometer/mge_group.py``.
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
__Model__ — Isothermal + ExternalShear lens, MGE source, 3 extra IsothermalSph galaxies.
"""
mass = af.Model(al.mp.Isothermal)
shear = af.Model(al.mp.ExternalShear)
lens = af.Model(al.Galaxy, redshift=0.5, mass=mass, shear=shear)

bulge = al.model_util.mge_model_from(
    mask_radius=mask_radius, total_gaussians=10, centre_prior_is_uniform=False
)
source = af.Model(al.Galaxy, redshift=1.0, bulge=bulge)

extra_galaxy_centres = [(0.5, 1.0), (-0.5, 1.5), (1.0, -0.5)]
extra_galaxies_list = []

for centre in extra_galaxy_centres:
    mass_extra = af.Model(al.mp.IsothermalSph)
    mass_extra.centre = centre
    mass_extra.einstein_radius = af.UniformPrior(lower_limit=0.0, upper_limit=0.3)
    extra_galaxy = af.Model(al.Galaxy, redshift=0.5, mass=mass_extra)
    extra_galaxies_list.append(extra_galaxy)

extra_galaxies = af.Collection(extra_galaxies_list)

model = af.Collection(
    galaxies=af.Collection(lens=lens, source=source),
    extra_galaxies=extra_galaxies,
)


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
