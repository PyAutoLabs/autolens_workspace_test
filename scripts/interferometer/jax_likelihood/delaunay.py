"""
Func Grad: Interferometer Delaunay Pixelization Source
=======================================================

This script tests if JAX can successfully compute the gradient of the log likelihood
of an `Interferometer` dataset with a model which uses a Delaunay pixelization source.

Mirrors `imaging/delaunay.py` but uses interferometer dataset loading and
`AnalysisInterferometer`. No apply_over_sampling — interferometer does not oversample.

__Env__

Test-harness configuration (PyAutoHands docs/env_profile_redesign.md §10).
JAX likelihood functions test JIT compilation; need JAX enabled and full-
size datasets.

ENV: jax full_datasets
"""

import numpy as np
import jax
import jax.numpy as jnp
from os import path

import autofit as af
import autolens as al
from autolens import conf

"""
__Mask__

We define the 'real_space_mask' which defines the grid the image the strong lens is
evaluated using.
"""
mask_radius = 3.0

real_space_mask = al.Mask2D.circular(
    shape_native=(256, 256),
    pixel_scales=0.1,
    radius=mask_radius,
)

"""
__Dataset__

Load the interferometer dataset from .fits files.
"""
dataset_name = "simple"
dataset_path = path.join("dataset", "interferometer", dataset_name)

"""
__Dataset Auto-Simulation__

If the dataset does not already exist on your system, it will be created by running
the corresponding simulator script.
"""
if al.util.dataset.should_simulate(dataset_path):
    import subprocess
    import sys

    subprocess.run(
        [
            sys.executable,
            "scripts/interferometer/simulator/simple.py",
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

print(f"Total Visiblities: {dataset.uv_wavelengths.shape[0]}")

"""
__Over Sampling__

Interferometer does not observe galaxies in a way where over sampling is necessary,
therefore all interferometer calculations are performed without over sampling.
"""

"""
__JAX & Preloads__

In JAX, calculations must use static shaped arrays with known and fixed indexes.
For the Delaunay pixelization, we compute the image-plane mesh grid before modeling
using the Hilbert image mesh, and append edge points to zero them.

- `pixels`: number of source pixels in the Delaunay mesh.
- `edge_pixels_total`: number of edge pixels zeroed in the source reconstruction.
"""
pixels = 750
edge_pixels_total = 30

# Use a Sersic image as adapt data (same as interferometer/rectangular.py) to avoid
# negative values in the dirty image causing NaN in pixel signal computation.
bulge_adapt = al.lp.Sersic()
adapt_image = bulge_adapt.image_2d_from(grid=dataset.grid)

galaxy_image_name_dict = {
    "('galaxies', 'lens')": adapt_image,
    "('galaxies', 'source')": adapt_image,
}

image_mesh = al.image_mesh.Hilbert(pixels=pixels, weight_power=3.5, weight_floor=0.01)

image_plane_mesh_grid = image_mesh.image_plane_mesh_grid_from(
    mask=real_space_mask,
    adapt_data=galaxy_image_name_dict["('galaxies', 'source')"],
)

image_plane_mesh_grid = al.image_mesh.append_with_circle_edge_points(
    image_plane_mesh_grid=image_plane_mesh_grid,
    centre=real_space_mask.mask_centre,
    radius=mask_radius + real_space_mask.pixel_scale / 2.0,
    n_points=edge_pixels_total,
)

total_mapper_pixels = image_plane_mesh_grid.shape[0]

adapt_images = al.AdaptImages(
    galaxy_name_image_dict=galaxy_image_name_dict,
    galaxy_name_image_plane_mesh_grid_dict={
        "('galaxies', 'source')": image_plane_mesh_grid
    },
)


"""
__Model__

We compose our model using `Model` objects, which represent the galaxies we fit to
our data. In this example we fit a model where:

 - The lens galaxy has an `Isothermal` mass and `ExternalShear`.
 - The source galaxy has a Delaunay pixelization.

"""
# Lens:

mass = af.Model(al.mp.PowerLaw)

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

# Source:

regularization = al.reg.AdaptSplit()

pixelization = af.Model(
    al.Pixelization,
    mesh=al.mesh.Delaunay(pixels=pixels, zeroed_pixels=edge_pixels_total),
    regularization=regularization,
)

source = af.Model(al.Galaxy, redshift=1.0, pixelization=pixelization)

# Overall Lens Model:

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

"""
The `info` attribute shows the model in a readable format.
"""
print(model.info)

"""
__Analysis__

The `AnalysisInterferometer` object defines the `log_likelihood_function` which will
be used to determine if JAX can compute its gradient.
"""
analysis = al.AnalysisInterferometer(
    dataset=dataset,
    adapt_images=adapt_images,
    raise_inversion_positions_likelihood_exception=False,
)

"""
The analysis and `log_likelihood_function` are internally wrapped into a `Fitness`
class in **PyAutoFit**, which pairs the model with likelihood.

This is the function on which JAX gradients are computed, so we create this class here.
"""
from autofit.non_linear.fitness import Fitness
import time

batch_size = 3

fitness = Fitness(
    model=model,
    analysis=analysis,
    fom_is_log_likelihood=True,
    resample_figure_of_merit=-1.0e99,
    batch_size=batch_size,
)

batch_size = fitness.batch_size

parameters = np.zeros((batch_size, model.total_free_parameters))

for i in range(batch_size):
    parameters[i, :] = model.physical_values_from_prior_medians

parameters = jnp.array(parameters)

start = time.time()
print()
print(fitness._vmap(parameters))
print("JAX Time To VMAP + JIT Function", time.time() - start)

start = time.time()
print()
result = fitness._vmap(parameters)
print(result)
print("JAX Time Taken using VMAP:", time.time() - start)
print("JAX Time Taken per Likelihood:", (time.time() - start) / batch_size)

np.testing.assert_allclose(
    np.array(result),
    -3165.42388511,
    rtol=1e-4,
    err_msg="interferometer/delaunay: JAX vmap likelihood mismatch",
)


"""
__Mass Sensitivity__

The literal above is evaluated at the model's prior medians, where a +5% change of
every lens mass parameter moves this likelihood by less than the literal's rtol
(audit 2026-08-06, autolens_workspace_test#253) — the literal alone would pass a
source-plane mass regression. This block pins mass sensitivity directly; the floor
is the audit-measured response divided by five (margin for platform drift).
"""
mass_indices = [
    i
    for i, name in enumerate(model.model_component_and_parameter_names)
    if ".mass." in name
    and "centre" not in name
    and "ell_comps" not in name
    and "redshift" not in name
]
assert (
    mass_indices
), "interferometer/delaunay: no mass parameters found for sensitivity check"

parameters_perturbed = np.array(model.physical_values_from_prior_medians)
for i in mass_indices:
    parameters_perturbed[i] *= 1.05

ll_median = float(np.asarray(result).ravel()[0])
ll_perturbed = float(
    np.asarray(fitness._vmap(jnp.array(parameters_perturbed[None, :]))).ravel()[0]
)
assert abs(ll_perturbed - ll_median) > 0.03, (
    f"interferometer/delaunay: likelihood insensitive to a +5% lens-mass perturbation "
    f"(median={ll_median}, perturbed={ll_perturbed}) — source-plane mass pipeline regression?"
)
print("PASS: mass-sensitivity floor exceeded.")

# The sparse inversion path must obey the same lane-isolation contract as
# imaging. Fitness sees the raw NaN after the forward pass and converts only
# that lane to its configured resample value.
poisoned_parameters = parameters.at[1, :].set(jnp.nan)
poisoned_result = np.asarray(fitness._vmap(poisoned_parameters))
finite_lanes = np.array([0, 2])

np.testing.assert_array_equal(
    poisoned_result[finite_lanes], np.asarray(result)[finite_lanes]
)
assert poisoned_result[1] == -1.0e99
print("PASS: poisoned sparse Delaunay vmap lane is isolated and resampled.")


"""
__Path A: jit-wrap ``analysis.fit_from``__
"""


instance = model.instance_from_prior_medians()

analysis_np = al.AnalysisInterferometer(
    dataset=dataset,
    adapt_images=adapt_images,
    raise_inversion_positions_likelihood_exception=False,
    use_jax=False,
)
fit_np = analysis_np.fit_from(instance=instance)
print("NumPy fit.log_likelihood:", float(fit_np.log_likelihood))

analysis_jit = al.AnalysisInterferometer(
    dataset=dataset,
    adapt_images=adapt_images,
    raise_inversion_positions_likelihood_exception=False,
    use_jax=True,
)
fit_jit_fn = jax.jit(analysis_jit.fit_from)
fit = fit_jit_fn(instance)

print("JIT fit.log_likelihood:", fit.log_likelihood)
assert isinstance(
    fit.log_likelihood, jnp.ndarray
), f"expected jax.Array, got {type(fit.log_likelihood)}"
np.testing.assert_allclose(
    float(fit.log_likelihood), float(fit_np.log_likelihood), rtol=1e-4
)
print("PASS: jit(fit_from) round-trip matches NumPy scalar.")

# Construct a valid instance first, then poison the downstream lens mapping. Profile
# validation intentionally rejects concrete NaN constructor inputs before fitting: an
# all-NaN free-parameter vector through `instance_from_vector` hands NaN `ell_comps` to
# autogalaxy's `validate_ell_comps` at `EllProfile.__init__`, which raises
# `exc.ModelParameterException` (a `FitException`, so a search resamples the point).
# The NaN therefore has to enter below the model-construction boundary, exactly as the
# `_vmap` lane above poisons the parameter array rather than an eager constructor.
nan_instance = model.instance_from_prior_medians()
nan_instance.galaxies.lens.mass.einstein_radius = np.nan
nan_fit = fit_jit_fn(nan_instance)
assert np.isnan(float(nan_fit.log_likelihood))
print("PASS: invalid Delaunay mesh reaches the raw interferometer likelihood as NaN.")


"""
__Path B: TransformerNUFFT cross-check__

Re-run the same vmap likelihood with the JAX-native nufftax-backed
TransformerNUFFT. Should match the TransformerDFT result because nufftax
agrees with the analytic DFT to ~1e-13 across the stress-tested
configurations. This proves the slow direct-DFT and fast NUFFT paths
produce the same end-to-end likelihood.
"""
dataset_nufft = al.Interferometer.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    uv_wavelengths_path=path.join(dataset_path, "uv_wavelengths.fits"),
    real_space_mask=real_space_mask,
    transformer_class=al.TransformerNUFFT,
)

analysis_nufft = al.AnalysisInterferometer(
    dataset=dataset_nufft,
    adapt_images=adapt_images,
    raise_inversion_positions_likelihood_exception=False,
)

fitness_nufft = Fitness(
    model=model,
    analysis=analysis_nufft,
    fom_is_log_likelihood=True,
    resample_figure_of_merit=-1.0e99,
)

# Path B's TransformerNUFFT JIT compilation peaks at ~15 GB on this corpus,
# OOM-killing python on 15 GB-RAM machines. The cross-check only needs one
# sample to prove NUFFT vs DFT agree, so restrict to parameters[:1] here.
result_nufft = fitness_nufft._vmap(parameters[:1])
print()
print("TransformerNUFFT vmap result:", result_nufft)

np.testing.assert_allclose(
    np.array(result_nufft),
    -3165.42388511,
    rtol=1e-4,
    err_msg="interferometer/delaunay: TransformerNUFFT vmap likelihood disagrees with TransformerDFT",
)
print("PASS: TransformerNUFFT cross-check matches TransformerDFT.")
