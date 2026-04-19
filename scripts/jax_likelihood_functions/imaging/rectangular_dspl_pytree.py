"""
Path A PoC: jit-wrap ``analysis.fit_from`` for DSPL + rectangular pixelizations
===============================================================================

Double-source-plane (DSPL) sibling of ``rectangular_pytree.py``. 3-plane
system: lens at z=0.5, intermediate lens+source at z=1.0, source at z=2.0.
Both the intermediate and the background source use rectangular pixelizations,
so the fit builds two separate inversions.

Drops the MGE bulge on the main lens (per
``admin_jammy/prompt/issued/fit_imaging_pytree_rectangular_dspl.md``) to keep
the variant unblocked on the ``linear_light_profile_intensity_dict`` issue.

What's likely to surface:
  - 3-plane ``Tracer`` pytree (``cosmology`` stays aux, but confirm no distance
    cache escapes into traced state).
  - Multi-pixelization ``galaxy_image_plane_mesh_grid_dict`` — this variant
    has 2 pixelized galaxies, testing the narrow fallback from
    PyAutoGalaxy#361.

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
__Dataset__ — same DSPL on-disk dataset used by ``rectangular_dspl.py``.
"""
sub_size = 4

dataset_path = path.join("dataset", "imaging", "jax_test_dspl")

if al.util.dataset.should_simulate(dataset_path):
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "scripts/jax_likelihood_functions/imaging/simulator_dspl.py"],
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

dataset = dataset.apply_over_sampling(
    over_sample_size_lp=over_sample_size,
    over_sample_size_pixelization=4,
)


"""
__Model__ — 3-plane DSPL: Isothermal lens_0 (z=0.5),
Isothermal + rectangular lens_1 (z=1.0), rectangular source (z=2.0).
No MGE bulge (avoids linear-light-profile visualization block).
"""
mesh_shape = (30, 30)

# Lens 0 (main deflector, z=0.5)
mass_0 = af.Model(al.mp.Isothermal)
mass_0.centre.centre_0 = af.UniformPrior(lower_limit=-0.2, upper_limit=0.2)
mass_0.centre.centre_1 = af.UniformPrior(lower_limit=-0.2, upper_limit=0.2)
mass_0.einstein_radius = af.UniformPrior(lower_limit=1.5, upper_limit=1.7)
mass_0.ell_comps.ell_comps_0 = af.UniformPrior(
    lower_limit=0.11111111111111108, upper_limit=0.1111111111111111
)
mass_0.ell_comps.ell_comps_1 = af.UniformPrior(lower_limit=-0.01, upper_limit=0.01)

shear = af.Model(al.mp.ExternalShear)
shear.gamma_1 = af.UniformPrior(lower_limit=-0.001, upper_limit=0.001)
shear.gamma_2 = af.UniformPrior(lower_limit=-0.001, upper_limit=0.001)

lens_0 = af.Model(al.Galaxy, redshift=0.5, mass=mass_0, shear=shear)

# Lens 1 (intermediate, z=1.0) — both deflector AND pixelized source
mass_1 = af.Model(al.mp.Isothermal)
mass_1.centre.centre_0 = af.UniformPrior(lower_limit=-0.2, upper_limit=0.2)
mass_1.centre.centre_1 = af.UniformPrior(lower_limit=-0.2, upper_limit=0.2)
mass_1.einstein_radius = af.UniformPrior(lower_limit=0.4, upper_limit=0.6)
mass_1.ell_comps.ell_comps_0 = af.UniformPrior(
    lower_limit=0.11111111111111108, upper_limit=0.1111111111111111
)
mass_1.ell_comps.ell_comps_1 = af.UniformPrior(lower_limit=-0.01, upper_limit=0.01)

mesh_1 = al.mesh.RectangularAdaptImage(shape=mesh_shape, weight_power=1.0)
pixelization_1 = al.Pixelization(mesh=mesh_1, regularization=al.reg.Adapt())

lens_1 = af.Model(al.Galaxy, redshift=1.0, mass=mass_1, pixelization=pixelization_1)

# Source (z=2.0) — pixelized
mesh_s = al.mesh.RectangularAdaptImage(shape=mesh_shape, weight_power=1.0)
pixelization_s = al.Pixelization(mesh=mesh_s, regularization=al.reg.Adapt())

source = af.Model(al.Galaxy, redshift=2.0, pixelization=pixelization_s)

model = af.Collection(
    galaxies=af.Collection(lens=lens_0, lens_1=lens_1, source=source)
)

galaxy_name_image_dict = {
    "('galaxies', 'lens')": dataset.data,
    "('galaxies', 'lens_1')": dataset.data,
    "('galaxies', 'source')": dataset.data,
}

adapt_images = al.AdaptImages(galaxy_name_image_dict=galaxy_name_image_dict)


"""
__Analysis__ on the JAX path.
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
__NumPy reference scalar__.
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
