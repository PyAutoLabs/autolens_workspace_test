"""
Path A PoC: jit-wrap ``analysis.fit_from`` for rectangular pixelization source
==============================================================================

Sibling of ``mge_pytree.py`` for the rectangular-pixelization source path
(see admin_jammy/prompt/issued/fit_imaging_pytree_rectangular.md). The source
``Galaxy`` carries a ``Pixelization(mesh=RectangularAdaptImage, regularization=Adapt)``,
driving the pytree cascade through ``Inversion``, ``Mapper*``, ``Mesh``, and
``Regularization`` types in the autoarray inversion layer.

Each run exposes the next unregistered type until the JIT path succeeds.

Success criterion:
  - ``jax.jit(analysis.fit_from)(instance)`` returns a ``FitImaging`` whose
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
__Dataset__ — same on-disk dataset used by ``rectangular.py``.
"""
sub_size = 4

dataset_path = path.join("dataset", "imaging", "jax_test")

if al.util.dataset.should_simulate(dataset_path):
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "scripts/jax_likelihood_functions/imaging/simulator.py"],
        check=True,
    )

dataset = al.Imaging.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    pixel_scales=0.2,
    over_sample_size_lp=sub_size,
    over_sample_size_pixelization=sub_size,
)

mask_radius = 3.5

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=mask_radius,
)

dataset = dataset.apply_mask(mask=mask)

over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=dataset.grid,
    sub_size_list=[4, 2, 1],
    radial_list=[0.3, 0.6],
    centre_list=[(0.0, 0.0)],
)

snr_no_lens = al.Array2D.from_fits(
    file_path=path.join(dataset_path, "snr_no_lens.fits"), pixel_scales=0.2
)

signal_to_noise_threshold = 3.0
over_sample_size_pixelization = np.where(
    snr_no_lens.native > signal_to_noise_threshold,
    4,
    2,
)
over_sample_size_pixelization = al.Array2D(
    values=over_sample_size_pixelization, mask=mask
)

dataset = dataset.apply_over_sampling(
    over_sample_size_lp=over_sample_size,
    over_sample_size_pixelization=over_sample_size_pixelization,
)


"""
__Model__ — Isothermal + ExternalShear lens, rectangular pixelization source with Adapt regularization.
"""
mesh_pixels_yx = 28
mesh_shape = (mesh_pixels_yx, mesh_pixels_yx)

mass = af.Model(al.mp.Isothermal)
mass.centre.centre_0 = af.UniformPrior(lower_limit=0.2, upper_limit=0.4)
mass.centre.centre_1 = af.UniformPrior(lower_limit=-0.4, upper_limit=-0.2)
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

mesh = al.mesh.RectangularAdaptImage(shape=mesh_shape, weight_power=1.0)
regularization = al.reg.Adapt()
pixelization = al.Pixelization(mesh=mesh, regularization=regularization)

source = af.Model(al.Galaxy, redshift=1.0, pixelization=pixelization)

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

galaxy_name_image_dict = {
    "('galaxies', 'lens')": dataset.data,
    "('galaxies', 'source')": dataset.data,
}

adapt_images = al.AdaptImages(galaxy_name_image_dict=galaxy_name_image_dict)


"""
__Analysis__ on the JAX path (``use_jax=True``) so ``fit_from`` builds with ``xp=jnp``.
"""
analysis = al.AnalysisImaging(
    dataset=dataset,
    adapt_images=adapt_images,
    raise_inversion_positions_likelihood_exception=False,
    settings=al.Settings(
        use_border_relocator=True,
        use_positive_only_solver=True,
        use_mixed_precision=True,
    ),
    use_jax=True,
)

register_model(model)

instance = model.instance_from_prior_medians()


"""
__NumPy reference scalar__ (cross-check once the JIT path returns).
"""
analysis_np = al.AnalysisImaging(
    dataset=dataset,
    adapt_images=adapt_images,
    raise_inversion_positions_likelihood_exception=False,
    settings=al.Settings(
        use_border_relocator=True,
        use_positive_only_solver=True,
        use_mixed_precision=True,
    ),
    use_jax=False,
)
fit_np = analysis_np.fit_from(instance=instance)
print("NumPy fit.log_likelihood:", float(fit_np.log_likelihood))


"""
__Path A: jit-wrap ``analysis.fit_from``__
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
