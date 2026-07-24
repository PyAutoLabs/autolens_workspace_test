"""
Modeling: Mass Total + Source Inversion
=======================================

This script fits an `Imaging` dataset of a 'galaxy-scale' strong lens with a model where:

 - The lens galaxy's light is omitted (and is not present in the simulated data).
 - The lens galaxy's total mass distribution is an `Isothermal` and `ExternalShear`.
 - The source galaxy's surface-brightness is an `Inversion`.

An `Inversion` reconstructs the source's light using a pixel-grid, which is regularized using a prior that forces
this reconstruction to be smooth. This uses `Pixelization`  objects and in this example we will
use their simplest forms, a `RectangularAdaptDensity` `Pixelization` and `Constant` `Regularization`.scheme.

Inversions are covered in detail in chapter 4 of the **HowToLens** lectures.

__Env__

Test-harness configuration (PyAutoHands docs/env_profile_redesign.md §10).
Beyond the bootstrapped dataset (which auto-simulates at any resolution), this
script runs inline padding/convolution sub-tests on fixed (51,51)/(21,21) grids
whose Mask2D construction the SMALL_DATASETS cap shrinks to 16x16, breaking the
array/mask shape assertion. Needs full resolution (mesh geometry, not committed
data).

ENV: full_datasets
"""

# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from os import path
from pathlib import Path

import autolens as al

"""
__Dataset__

Load and plot the strong lens dataset `mass_sie__source_sersic` via .fits files, which we will fit with the lens model.
"""
dataset_label = "build"
dataset_type = "imaging"
dataset_name = "with_lens_light"

dataset_path = path.join("dataset", dataset_label, dataset_type, dataset_name)

"""
__Dataset Auto-Simulation__

If the dataset does not already exist on your system, it will be created by running the corresponding
simulator script. This ensures that all example scripts can be run without manually simulating data first.
"""
if al.util.dataset.should_simulate(dataset_path):
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "scripts/imaging/simulator/with_lens_light.py"],
        check=True,
    )

dataset = al.Imaging.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    pixel_scales=0.2,
)

"""
__Mask__

The model-fit requires a 2D mask defining the regions of the image we fit the lens model to the data, which we define
and use to set up the `Imaging` object that the lens model fits.
"""
mask_radius = 7.2

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=mask_radius,
)

dataset = dataset.apply_mask(mask=mask)

"""
Setup the lens galaxy's mass (SIE+Shear) and source galaxy light (elliptical Sersic) for this simulated lens.

For lens modeling, defining ellipticity in terms of the `ell_comps` improves the model-fitting procedure.

However, for simulating a strong lens you may find it more intuitive to define the elliptical geometry using the 
axis-ratio of the profile (axis_ratio = semi-major axis / semi-minor axis = b/a) and position angle, where angle is
in degrees and defined counter clockwise from the positive x-axis.

We can use the `convert` module to determine the elliptical components from the axis-ratio and angle.
"""
lens_galaxy = al.Galaxy(
    redshift=0.5,
    bulge=al.lp.DevVaucouleursSph(
        centre=(0.0, 0.0),
        intensity=0.1,
        effective_radius=0.8,
    ),
    mass=al.mp.IsothermalSph(
        centre=(0.0, 0.0),
        einstein_radius=1.6,
    ),
)

source_galaxy = al.Galaxy(
    redshift=1.0,
    bulge=al.lp.ExponentialSph(
        centre=(0.0, 0.1),
        intensity=0.3,
        effective_radius=0.1,
    ),
)

"""
Use these galaxies to setup a tracer, which will generate the image for the simulated `Imaging` dataset.
"""
tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

image_2d = tracer.image_2d_from(grid=dataset.grid)

blurring_image_2d = tracer.image_2d_from(
    grid=dataset.grids.blurring,
)


via_fft = dataset.psf.convolved_image_from(
    image=image_2d, blurring_image=blurring_image_2d, xp=jnp
)


via_real_space = dataset.psf.convolved_image_via_real_space_np_from(
    image=image_2d, blurring_image=blurring_image_2d, xp=np
)

residuals = via_fft.native - via_real_space.native

script_path = Path("scripts") / "imaging" / "images"
script_path.mkdir(parents=True, exist_ok=True)

print(f"Max residual = {residuals.max()}")
print(
    f"Max residual located at {jnp.unravel_index(jnp.argmax(residuals.array), residuals.array.shape)}"
)

plt.imshow(residuals.array, cmap="viridis")
plt.colorbar()
plt.title("Residuals between FFT and Real Space Convolution")
plt.xlabel("X Pixel")
plt.ylabel("Y Pixel")
plt.savefig(script_path / "residuals.png", dpi=300)

mapping_matrix = np.zeros((image_2d.shape[0], 2))

mapping_matrix[:, 0] = image_2d
mapping_matrix[:, 1] = image_2d + 1


via_fft = dataset.psf.convolved_mapping_matrix_from(
    mapping_matrix=mapping_matrix, mask=image_2d.mask, xp=jnp
)


via_real_space = dataset.psf.convolved_mapping_matrix_via_real_space_np_from(
    mapping_matrix=mapping_matrix, mask=image_2d.mask, xp=np
)

residuals = via_fft - via_real_space

print(f"Mapping Matrix Max residual = {residuals.max()}")


"""
__Mask Padding__

When the mask is close to the image edge and the PSF kernel footprint extends
beyond the boundary, the blurring mask is automatically padded. This test
verifies that the padded convolution produces the same result as an equivalent
centred configuration that requires no padding.

We simulate a compact lens+source on two grids:
  - A large 51x51 grid with the model centred → no padding needed.
  - A small 21x21 grid with the model offset near the edge → padding triggered.

Both should give identical log-likelihoods and chi_squared ≈ 0.
"""

import warnings

pixel_scales = 0.2

psf_pad = al.Convolver.from_gaussian(
    shape_native=(11, 11), pixel_scales=pixel_scales, sigma=0.75, normalize=True
)

lens_centred = al.Galaxy(
    redshift=0.5,
    light=al.lp.Sersic(
        centre=(0.0, 0.0),
        intensity=0.1,
        effective_radius=0.3,
        sersic_index=2.0,
    ),
    mass=al.mp.Isothermal(centre=(0.0, 0.0), einstein_radius=1.0),
)
source_centred = al.Galaxy(
    redshift=1.0,
    light=al.lp.Exponential(
        centre=(0.0, 0.0),
        intensity=0.3,
        effective_radius=0.2,
    ),
)
tracer_centred = al.Tracer(galaxies=[lens_centred, source_centred])

sim_pad = al.SimulatorImaging(
    exposure_time=300.0,
    psf=psf_pad,
    add_poisson_noise_to_data=False,
)

# --- Centred fit on a large grid: no padding needed ---
grid_large = al.Grid2D.uniform(
    shape_native=(51, 51),
    pixel_scales=pixel_scales,
    over_sample_size=1,
)
dataset_centred = sim_pad.via_tracer_from(tracer=tracer_centred, grid=grid_large)
dataset_centred.noise_map = al.Array2D.ones(
    shape_native=(51, 51),
    pixel_scales=pixel_scales,
)
mask_centred = al.Mask2D.circular(
    shape_native=(51, 51),
    pixel_scales=pixel_scales,
    radius=0.6,
    centre=(0.0, 0.0),
)
masked_centred = dataset_centred.apply_mask(mask=mask_centred)
fit_centred = al.FitImaging(dataset=masked_centred, tracer=tracer_centred)

# --- Off-centre fit on a small grid: triggers padding ---
offset = (0.0, 1.2)
lens_off = al.Galaxy(
    redshift=0.5,
    light=al.lp.Sersic(
        centre=offset,
        intensity=0.1,
        effective_radius=0.3,
        sersic_index=2.0,
    ),
    mass=al.mp.Isothermal(centre=offset, einstein_radius=1.0),
)
source_off = al.Galaxy(
    redshift=1.0,
    light=al.lp.Exponential(
        centre=offset,
        intensity=0.3,
        effective_radius=0.2,
    ),
)
tracer_off = al.Tracer(galaxies=[lens_off, source_off])

grid_small = al.Grid2D.uniform(
    shape_native=(21, 21),
    pixel_scales=pixel_scales,
    over_sample_size=1,
)
dataset_off = sim_pad.via_tracer_from(tracer=tracer_off, grid=grid_small)
dataset_off.noise_map = al.Array2D.ones(
    shape_native=(21, 21),
    pixel_scales=pixel_scales,
)
mask_off = al.Mask2D.circular(
    shape_native=(21, 21),
    pixel_scales=pixel_scales,
    radius=0.6,
    centre=offset,
)

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    masked_off = dataset_off.apply_mask(mask=mask_off)
    fit_off = al.FitImaging(dataset=masked_off, tracer=tracer_off)
    padding_occurred = any("Mask padded" in str(x.message) for x in w)

assert padding_occurred, "Expected mask padding to be triggered for the off-centre mask"
assert (
    fit_centred.chi_squared < 1e-4
), f"Centred chi_squared too large: {fit_centred.chi_squared}"
assert (
    fit_off.chi_squared < 1e-4
), f"Off-centre chi_squared too large: {fit_off.chi_squared}"

likelihood_diff = abs(fit_centred.log_likelihood - fit_off.log_likelihood)
assert (
    likelihood_diff < 1e-4
), f"Padded and non-padded log-likelihoods differ by {likelihood_diff}"

print(f"\nMask padding test PASSED")
print(
    f"  Centred  log_likelihood = {fit_centred.log_likelihood:.8f}  chi_squared = {fit_centred.chi_squared:.8f}"
)
print(
    f"  Padded   log_likelihood = {fit_off.log_likelihood:.8f}  chi_squared = {fit_off.chi_squared:.8f}"
)
print(f"  Difference = {likelihood_diff:.2e}")

"""
__Oversampled PSF: FFT vs Real Space__

An oversampled PSF (`convolve_over_sample_size > 1`, supplied at a multiple of the image resolution) runs
convolution on the upscaled grid and bins back to image resolution. The JAX path uses the FFT formalism and the
numpy path direct real-space convolution — as with the s=1 checks above, the two must agree.

Full numerical coverage (ground-truth values, every supported model surface through `FitImaging`, guards) is in
`convolution_over_sampled.py`.
"""
s = 2

kernel_fine_n = 11
kc = (np.arange(kernel_fine_n) - (kernel_fine_n - 1) / 2.0) * (0.2 / s)
kyy, kxx = np.meshgrid(-kc, kc, indexing="ij")
kernel_fine = np.exp(-0.5 * (kyy**2 + kxx**2) / 0.15**2)

psf_over = al.Convolver(
    kernel=al.Array2D.no_mask(values=kernel_fine, pixel_scales=0.2 / s),
    normalize=True,
    convolve_over_sample_size=s,
)

mask_over = al.Mask2D.circular(shape_native=(21, 21), pixel_scales=0.2, radius=1.8)
grid_over = al.Grid2D.from_mask(mask=mask_over, over_sample_size=s)
blurring_mask_over = mask_over.derive_mask.blurring_from(
    kernel_shape_native=psf_over.kernel_shape_image_resolution, allow_padding=True
)
blurring_grid_over = al.Grid2D.from_mask(mask=blurring_mask_over, over_sample_size=s)

image_sub = tracer_centred.image_2d_from(grid=grid_over.over_sampled)
blurring_sub = tracer_centred.image_2d_from(grid=blurring_grid_over.over_sampled)

via_fft_over = psf_over.convolved_image_from(
    image=jnp.asarray(np.array(image_sub)),
    blurring_image=jnp.asarray(np.array(blurring_sub)),
    mask=mask_over,
    xp=jnp,
)

via_real_space_over = psf_over.convolved_image_from(
    image=np.array(image_sub),
    blurring_image=np.array(blurring_sub),
    mask=mask_over,
    xp=np,
)

residual_over = np.max(np.abs(np.array(via_fft_over) - np.array(via_real_space_over)))

print(f"\nOversampled (s={s}) FFT vs real-space max residual = {residual_over:.3e}")

assert (
    residual_over < 1.0e-8
), f"Oversampled FFT and real-space convolution disagree: {residual_over}"

print("Oversampled FFT vs real-space parity PASSED")
