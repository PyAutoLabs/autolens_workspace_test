"""
Func Grad: Interferometer Rectangular Pixelization via JAX Sparse NUFFT Operator
==================================================================================

This script tests if JAX can successfully compute the gradient of the log likelihood
of an `Interferometer` dataset using the JAX sparse-operator NUFFT path.

This is a copy of `interferometer/rectangular.py` with one additional line after
`Interferometer.from_fits(...)`:

    dataset = dataset.apply_sparse_operator(use_jax=True, show_progress=True)

The sparse NUFFT operator is aux state and must NOT be constructed inside the JIT
trace. It is built once before analysis construction and carried as static state.

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

Load and plot the galaxy dataset `operated` via .fits files, which we will fit with
the model.
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

"""
__Sparse Operator__

Apply the JAX sparse NUFFT operator. This builds the sparse operator once as aux
state — it must NOT be constructed inside the JIT trace.
"""
dataset = dataset.apply_sparse_operator(use_jax=True, show_progress=True)

print(f"Total Visiblities: {dataset.uv_wavelengths.shape[0]}")

"""
__Positions__
"""
positions = al.Grid2DIrregular(
    al.from_json(file_path=path.join(dataset_path, "positions.json"))
)

"""
__Over Sampling__

Interferometer does not observe galaxies in a way where over sampling is necessary,
therefore all interferometer calculations are performed without over sampling.

__Mesh Shape__

The `mesh_shape` parameter defines number of pixels used by the rectangular mesh to
reconstruct the source, set below to 8 x 8.
"""
mesh_pixels_yx = 8
mesh_shape = (mesh_pixels_yx, mesh_pixels_yx)

"""
__Model__

We compose our model using `Model` objects, which represent the galaxies we fit to
our data.
"""
# Lens:

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

# Source:

mesh = al.mesh.RectangularRTUAdaptDensity(shape=mesh_shape)

regularization = al.reg.Adapt()

pixelization = al.Pixelization(mesh=mesh, regularization=regularization)

source = af.Model(al.Galaxy, redshift=1.0, pixelization=pixelization)

# Overall Lens Model:

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))


bulge = al.lp.Sersic()

image = bulge.image_2d_from(grid=dataset.grid)

galaxy_name_image_dict = {
    "('galaxies', 'lens')": image,
    "('galaxies', 'source')": image,
}


adapt_images = al.AdaptImages(galaxy_name_image_dict=galaxy_name_image_dict)

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
"""
from autofit.non_linear.fitness import Fitness
import time

batch_size = 1

fitness = Fitness(
    model=model,
    analysis=analysis,
    fom_is_log_likelihood=True,
    resample_figure_of_merit=-1.0e99,
)

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
    -3164.286252,
    rtol=1e-4,
    err_msg="interferometer/rectangular_sparse: JAX vmap likelihood mismatch",
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
), "interferometer/rectangular_sparse: no mass parameters found for sensitivity check"

parameters_perturbed = np.array(model.physical_values_from_prior_medians)
for i in mass_indices:
    parameters_perturbed[i] *= 1.05

ll_median = float(np.asarray(result).ravel()[0])
ll_perturbed = float(
    np.asarray(fitness._vmap(jnp.array(parameters_perturbed[None, :]))).ravel()[0]
)
assert abs(ll_perturbed - ll_median) > 0.008, (
    f"interferometer/rectangular_sparse: likelihood insensitive to a +5% lens-mass perturbation "
    f"(median={ll_median}, perturbed={ll_perturbed}) — source-plane mass pipeline regression?"
)
print("PASS: mass-sensitivity floor exceeded.")


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


"""
__Path B: TransformerDFT, no sparse operator__

The Path A run above uses TransformerDFT + `apply_sparse_operator(use_jax=True)`
(the cached-precision-matrix accelerator for pixelization). This pass uses
the same TransformerDFT but skips the sparse-operator optimization — the
plain direct-DFT pixelization path. After the Pmax > 1 / extent-indexing fix
(issue #314), the two paths agree to numerical precision: the sparse-operator
precomputation is mathematically exact, not a "numerical reformulation".
Path B's literal therefore matches Path A and
`scripts/interferometer/rectangular.py` (the same
model + DFT-no-sparse path).
"""
dataset_dft_nosparse = al.Interferometer.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    uv_wavelengths_path=path.join(dataset_path, "uv_wavelengths.fits"),
    real_space_mask=real_space_mask,
    transformer_class=al.TransformerDFT,
)

analysis_dft_nosparse = al.AnalysisInterferometer(
    dataset=dataset_dft_nosparse,
    adapt_images=adapt_images,
    raise_inversion_positions_likelihood_exception=False,
)

fitness_dft_nosparse = Fitness(
    model=model,
    analysis=analysis_dft_nosparse,
    fom_is_log_likelihood=True,
    resample_figure_of_merit=-1.0e99,
)

result_dft_nosparse = fitness_dft_nosparse._vmap(parameters)
print()
print("TransformerDFT (no sparse) vmap result:", result_dft_nosparse)

np.testing.assert_allclose(
    np.array(result_dft_nosparse),
    -3164.286252,  # matches rectangular.py (same model, same DFT-no-sparse path)
    rtol=1e-4,
    err_msg="interferometer/rectangular_sparse: DFT-no-sparse vmap likelihood disagrees with rectangular.py reference",
)
print("PASS: TransformerDFT (no sparse) matches rectangular.py canonical likelihood.")


"""
__Path C: TransformerNUFFT (no sparse operator)__

TransformerNUFFT is not used with `apply_sparse_operator` here. The reason
originally recorded was specific to the legacy pynufft backend's
kernel-deconvolved adjoint scale; that backend has since been removed, so
whether the incompatibility still holds against the nufftax adjoint has not
been re-verified — this path deliberately does not depend on the answer.
Run plain TransformerNUFFT + direct forward NUFFT for the pixelization.
Should match Path B (DFT-no-sparse) since nufftax matches the analytic DFT
to ~1e-13 in the forward operator.
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

result_nufft = fitness_nufft._vmap(parameters)
print()
print("TransformerNUFFT vmap result:", result_nufft)

np.testing.assert_allclose(
    np.array(result_nufft),
    -3164.286252,  # matches DFT-no-sparse path (Path B)
    rtol=1e-4,
    err_msg="interferometer/rectangular_sparse: TransformerNUFFT vmap likelihood disagrees with DFT-no-sparse",
)
print("PASS: TransformerNUFFT cross-check matches TransformerDFT (no sparse).")
