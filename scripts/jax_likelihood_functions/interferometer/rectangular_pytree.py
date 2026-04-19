"""
Path A PoC: jit-wrap ``analysis.fit_from`` for interferometer + rectangular pixelization
=========================================================================================

Intersection of two already-shipped siblings:

- ``interferometer/mge_pytree.py`` — ``FitInterferometer`` + ``TransformerDFT`` +
  pytree registrations for ``Interferometer`` / ``Visibilities`` / transformers.
- ``imaging/rectangular_pytree.py`` — rectangular pixelization source pytree
  cascade through ``Inversion``, ``Mapper*``, ``Mesh``, ``Regularization``.

Model:
  - Lens: Isothermal + ExternalShear (z=0.5)
  - Source: ``Pixelization(mesh=RectangularAdaptDensity(8, 8), regularization=Adapt)`` (z=1.0)

Task prompt (``admin_jammy/prompt/issued/fit_interferometer_pytree_rectangular.md``):
canary for the transformer × inversion interaction under JIT. If both predecessors
landed clean, this should pass with no new registrations.

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
__Dataset__ — same on-disk dataset used by ``interferometer/rectangular.py``.
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
__Model__ — Isothermal + ExternalShear lens, rectangular pixelization source with Adapt regularization.
"""
mesh_pixels_yx = 8
mesh_shape = (mesh_pixels_yx, mesh_pixels_yx)

mass = af.Model(al.mp.Isothermal)
mass.centre.centre_0 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
mass.centre.centre_1 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
mass.einstein_radius = af.UniformPrior(lower_limit=1.5, upper_limit=1.7)
mass.ell_comps.ell_comps_0 = af.UniformPrior(
    lower_limit=0.11111111111111108, upper_limit=0.1111111111111111
)
mass.ell_comps.ell_comps_1 = af.UniformPrior(lower_limit=-0.01, upper_limit=0.01)

shear = af.Model(al.mp.ExternalShear)
shear.gamma_1 = af.UniformPrior(lower_limit=-0.001, upper_limit=0.001)
shear.gamma_2 = af.UniformPrior(lower_limit=-0.001, upper_limit=0.001)

lens = af.Model(
    al.Galaxy,
    redshift=0.5,
    mass=mass,
    shear=shear,
)

mesh = al.mesh.RectangularAdaptDensity(shape=mesh_shape)
regularization = al.reg.Adapt()
pixelization = al.Pixelization(mesh=mesh, regularization=regularization)

source = af.Model(al.Galaxy, redshift=1.0, pixelization=pixelization)

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))


"""
__Adapt images__ — Sersic-generated images for both lens and source, matching the reference script.
"""
bulge = al.lp.Sersic()
adapt_image = bulge.image_2d_from(grid=dataset.grid)

galaxy_name_image_dict = {
    "('galaxies', 'lens')": adapt_image,
    "('galaxies', 'source')": adapt_image,
}

adapt_images = al.AdaptImages(galaxy_name_image_dict=galaxy_name_image_dict)


"""
__Analysis__ on the JAX path (``use_jax=True``).
"""
analysis = al.AnalysisInterferometer(
    dataset=dataset,
    adapt_images=adapt_images,
    raise_inversion_positions_likelihood_exception=False,
    use_jax=True,
)

register_model(model)

instance = model.instance_from_prior_medians()


"""
__NumPy reference scalar__.
"""
analysis_np = al.AnalysisInterferometer(
    dataset=dataset,
    adapt_images=adapt_images,
    raise_inversion_positions_likelihood_exception=False,
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
