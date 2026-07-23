"""
Convolution: Oversampled PSF
============================

Numerical tests of oversampled PSF convolution (`convolve_over_sample_size > 1`), where the PSF is supplied at a
multiple of the image resolution, images are evaluated on the over-sampled grid, convolution runs at the fine
resolution and the result is binned back to image resolution.

Covers, at `convolve_over_sample_size=2` with s=1 parity checks:

 1. The brute-force ground-truth reference values of the design phase (PyAutoArray#353).
 2. Full `FitImaging` self-consistency for every supported model surface:
    standard light profiles, linear light profiles, operated light profiles and pixelized sources
    (mapping formalism).
 3. The loud guards on unsupported surfaces: the sparse formalism, adaptive over sampling and
    mismatched over-sample sizes.

Library support shipped in PyAutoArray#355 (Convolver + dataset API), PyAutoArray#357 (inversion mapping
formalism) and PyAutoGalaxy#481 (operate/image consumer + linear light profiles).
"""
# ENV: full_datasets
# Reads pre-committed full-resolution data; the SMALL_DATASETS 15x15
# cap would break the mask/shape assertion.

# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import numpy as np

import autolens as al

"""
__Ground Truth Reference__

The design phase produced an independent brute-force implementation of oversampled convolution
(`PyAutoMind/feature/autoarray/oversampling_ground_truth.py`): an 11x11 image at 1.0" pixels, circular mask of
radius 3.5" (37 pixels), an off-centre Gaussian source (sigma=1.2", centre (0.3, -0.4)") and a Gaussian PSF
(sigma=0.8") with a fixed physical kernel radius of 2.0". Its reference values are pinned here through the
public API, at s=1 (parity with the existing Convolver) and s=2.
"""


def gaussian_2d(y, x, sigma, centre=(0.0, 0.0)):
    r2 = (y - centre[0]) ** 2 + (x - centre[1]) ** 2
    return (1.0 / (sigma * np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * r2 / sigma**2)


def gaussian_kernel(n, pixel_scale, sigma):
    c = (np.arange(n) - (n - 1) / 2.0) * pixel_scale
    yy, xx = np.meshgrid(-c, c, indexing="ij")
    return gaussian_2d(yy, xx, sigma)


mask_gt = al.Mask2D.circular(shape_native=(11, 11), pixel_scales=1.0, radius=3.5)

for s, kernel_n, expected in [
    (
        1,
        5,
        {
            "sum": 2.807349652595196e00,
            0: 3.655472905370449e-02,
            17: 2.069771979137382e-01,
            36: 1.042470837248629e-02,
        },
    ),
    (
        2,
        9,
        {
            "sum": 2.796562184524787e00,
            0: 3.726289901353439e-02,
            17: 2.025075336159483e-01,
            36: 1.090767109119494e-02,
        },
    ),
]:
    kernel = al.Array2D.no_mask(
        values=gaussian_kernel(n=kernel_n, pixel_scale=1.0 / s, sigma=0.8),
        pixel_scales=1.0 / s,
    )
    psf = al.Convolver(kernel=kernel, normalize=True, convolve_over_sample_size=s)

    grid = al.Grid2D.from_mask(mask=mask_gt, over_sample_size=s)
    blurring_mask = mask_gt.derive_mask.blurring_from(
        kernel_shape_native=psf.kernel_shape_image_resolution, allow_padding=True
    )
    blurring_grid = al.Grid2D.from_mask(mask=blurring_mask, over_sample_size=s)

    def source_on(grid_like):
        arr = np.array(grid_like)
        return gaussian_2d(arr[:, 0], arr[:, 1], 1.2, (0.3, -0.4))

    if s == 1:
        image = al.Array2D(values=source_on(grid), mask=mask_gt)
        blurring_image = al.Array2D(values=source_on(blurring_grid), mask=blurring_mask)
        convolved = psf.convolved_image_via_real_space_np_from(
            image=image, blurring_image=blurring_image
        )
    else:
        convolved = psf.convolved_image_from(
            image=source_on(grid.over_sampled),
            blurring_image=source_on(blurring_grid.over_sampled),
            mask=mask_gt,
        )

    convolved = np.array(convolved)

    assert abs(convolved.sum() - expected["sum"]) < 1.0e-12
    for idx in (0, 17, 36):
        assert abs(convolved[idx] - expected[idx]) < 1.0e-12

print("Ground-truth reference values PASSED (s=1 parity + s=2)")

"""
__FitImaging Setup__

Every supported model surface is now fitted end-to-end. The observed data is built through the (unit-tested)
oversampled blurred-image machinery itself, so a fit of the same model at s=2 must give chi_squared ~ 0 —
while a fit of the same data with the equivalent image-resolution (s=1) PSF leaves measurable residuals,
demonstrating that the oversampling changes the model image.
"""
pixel_scales = 0.2
s = 2

mask = al.Mask2D.circular(shape_native=(21, 21), pixel_scales=pixel_scales, radius=1.8)

psf_fine = al.Convolver(
    kernel=al.Array2D.no_mask(
        values=gaussian_kernel(n=11, pixel_scale=pixel_scales / s, sigma=0.15),
        pixel_scales=pixel_scales / s,
    ),
    normalize=True,
    convolve_over_sample_size=s,
)

psf_native = al.Convolver(
    kernel=al.Array2D.no_mask(
        values=gaussian_kernel(n=7, pixel_scale=pixel_scales, sigma=0.15),
        pixel_scales=pixel_scales,
    ),
    normalize=True,
)


def dataset_from(tracer, psf, convolve_over_sample_size):
    """
    Build an `Imaging` dataset whose data is the tracer's blurred image computed with the input PSF —
    self-consistent by construction, so fitting the same tracer with the same PSF gives chi_squared ~ 0.
    """
    dataset = al.Imaging(
        data=al.Array2D.no_mask(values=np.zeros((21, 21)), pixel_scales=pixel_scales),
        noise_map=al.Array2D.no_mask(
            values=np.ones((21, 21)), pixel_scales=pixel_scales
        ),
        psf=psf,
        over_sample_size_lp=(
            convolve_over_sample_size if convolve_over_sample_size > 1 else 4
        ),
        over_sample_size_pixelization=(
            convolve_over_sample_size if convolve_over_sample_size > 1 else 4
        ),
        convolve_over_sample_size_lp=convolve_over_sample_size,
        convolve_over_sample_size_pixelization=convolve_over_sample_size,
    ).apply_mask(mask=mask)

    blurred = tracer.blurred_image_2d_from(
        grid=dataset.grids.lp, blurring_grid=dataset.grids.blurring, psf=dataset.psf
    )

    return al.Imaging(
        data=al.Array2D(values=np.array(blurred), mask=mask).native,
        noise_map=al.Array2D.no_mask(
            values=np.ones((21, 21)), pixel_scales=pixel_scales
        ),
        psf=psf,
        over_sample_size_lp=(
            convolve_over_sample_size if convolve_over_sample_size > 1 else 4
        ),
        over_sample_size_pixelization=(
            convolve_over_sample_size if convolve_over_sample_size > 1 else 4
        ),
        convolve_over_sample_size_lp=convolve_over_sample_size,
        convolve_over_sample_size_pixelization=convolve_over_sample_size,
    ).apply_mask(mask=mask)


lens_mass = al.mp.Isothermal(centre=(0.0, 0.0), einstein_radius=1.0)

"""
__Surface 1: Standard Light Profiles__
"""
tracer_lp = al.Tracer(
    galaxies=[
        al.Galaxy(
            redshift=0.5,
            bulge=al.lp.Sersic(
                centre=(0.0, 0.0),
                intensity=0.5,
                effective_radius=0.4,
                sersic_index=2.0,
            ),
            mass=lens_mass,
        ),
        al.Galaxy(
            redshift=1.0,
            bulge=al.lp.Exponential(
                centre=(0.05, 0.05), intensity=0.3, effective_radius=0.2
            ),
        ),
    ]
)

dataset_s2 = dataset_from(tracer=tracer_lp, psf=psf_fine, convolve_over_sample_size=s)
fit_s2 = al.FitImaging(dataset=dataset_s2, tracer=tracer_lp)

assert fit_s2.chi_squared < 1.0e-8, f"lp s=2 chi_squared = {fit_s2.chi_squared}"

# The same data fitted with the image-resolution PSF leaves real residuals.
dataset_s1 = al.Imaging(
    data=dataset_s2.data.native,
    noise_map=al.Array2D.no_mask(values=np.ones((21, 21)), pixel_scales=pixel_scales),
    psf=psf_native,
).apply_mask(mask=mask)
fit_s1 = al.FitImaging(dataset=dataset_s1, tracer=tracer_lp)

assert fit_s1.chi_squared > 1.0e3 * max(fit_s2.chi_squared, 1.0e-12), (
    f"expected s=1 fit of s=2 data to be measurably worse: "
    f"s1={fit_s1.chi_squared}, s2={fit_s2.chi_squared}"
)

print(
    f"Standard light profiles PASSED  (chi2 s=2: {fit_s2.chi_squared:.3e}, s=1: {fit_s1.chi_squared:.3e})"
)

"""
__Surface 2: Operated Light Profiles__

Operated profiles represent already-convolved components (e.g. a point source fitted with a Gaussian);
they are added at image resolution without further blurring, alongside blurred standard components.
"""
tracer_operated = al.Tracer(
    galaxies=[
        al.Galaxy(
            redshift=0.5,
            bulge=al.lp.Sersic(
                centre=(0.0, 0.0),
                intensity=0.5,
                effective_radius=0.4,
                sersic_index=2.0,
            ),
            psf_component=al.lp_operated.Gaussian(
                centre=(0.0, 0.0), intensity=0.2, sigma=0.3
            ),
            mass=lens_mass,
        ),
        al.Galaxy(
            redshift=1.0,
            bulge=al.lp.Exponential(
                centre=(0.05, 0.05), intensity=0.3, effective_radius=0.2
            ),
        ),
    ]
)

dataset_operated = dataset_from(
    tracer=tracer_operated, psf=psf_fine, convolve_over_sample_size=s
)
fit_operated = al.FitImaging(dataset=dataset_operated, tracer=tracer_operated)

assert (
    fit_operated.chi_squared < 1.0e-8
), f"operated s=2 chi_squared = {fit_operated.chi_squared}"

print(f"Operated light profiles PASSED  (chi2 s=2: {fit_operated.chi_squared:.3e})")

"""
__Surface 3: Linear Light Profiles__

The data is generated with standard profiles of known intensity; the linear fit must recover those
intensities via the fine-resolution `operated_mapping_matrix_override` path.
"""
tracer_linear = al.Tracer(
    galaxies=[
        al.Galaxy(
            redshift=0.5,
            bulge=al.lp_linear.Sersic(
                centre=(0.0, 0.0), effective_radius=0.4, sersic_index=2.0
            ),
            mass=lens_mass,
        ),
        al.Galaxy(
            redshift=1.0,
            bulge=al.lp_linear.Exponential(centre=(0.05, 0.05), effective_radius=0.2),
        ),
    ]
)

fit_linear = al.FitImaging(dataset=dataset_s2, tracer=tracer_linear)

intensities = np.array(fit_linear.inversion.reconstruction)

assert (
    fit_linear.chi_squared < 1.0e-6
), f"linear s=2 chi_squared = {fit_linear.chi_squared}"
assert np.allclose(
    np.sort(intensities), np.sort(np.array([0.5, 0.3])), atol=1.0e-4
), f"linear intensities not recovered: {intensities}"

print(
    f"Linear light profiles PASSED  (chi2 s=2: {fit_linear.chi_squared:.3e}, "
    f"intensities: {np.sort(intensities)})"
)

"""
__Surface 4: Pixelized Source (Mapping Formalism)__

A rectangular pixelization reconstructs the source through the oversampled inversion wiring
(sub-resolution mapping matrix, fine convolution, mean bin-down). The reconstruction cannot be exact,
so the assertions are that the fit runs, is finite, and reconstructs the data well.
"""
pixelization = al.Pixelization(
    mesh=al.mesh.RectangularUniform(shape=(20, 20)),
    regularization=al.reg.Constant(coefficient=1.0e-4),
)

tracer_pix = al.Tracer(
    galaxies=[
        al.Galaxy(redshift=0.5, mass=lens_mass),
        al.Galaxy(redshift=1.0, pixelization=pixelization),
    ]
)

# Source-only data (no lens light) keeps the inversion self-contained.
tracer_src_only = al.Tracer(
    galaxies=[
        al.Galaxy(redshift=0.5, mass=lens_mass),
        al.Galaxy(
            redshift=1.0,
            bulge=al.lp.Exponential(
                centre=(0.05, 0.05), intensity=0.3, effective_radius=0.2
            ),
        ),
    ]
)

dataset_pix = dataset_from(
    tracer=tracer_src_only, psf=psf_fine, convolve_over_sample_size=s
)
fit_pix = al.FitImaging(dataset=dataset_pix, tracer=tracer_pix)

assert np.isfinite(fit_pix.log_evidence), "pixelized s=2 log_evidence not finite"
assert (
    fit_pix.chi_squared < 1.0e-1
), f"pixelized s=2 reconstruction poor: chi_squared = {fit_pix.chi_squared}"

print(
    f"Pixelized source (mapping formalism) PASSED  "
    f"(chi2 s=2: {fit_pix.chi_squared:.3e}, log_evidence: {fit_pix.log_evidence:.2f})"
)

"""
__Guards__

The unsupported surfaces must raise loudly rather than silently degrade.
"""
raised = 0

try:
    dataset_pix.apply_sparse_operator()
except al.exc.DatasetException:
    raised += 1

try:
    al.Imaging(
        data=al.Array2D.no_mask(values=np.zeros((21, 21)), pixel_scales=pixel_scales),
        noise_map=al.Array2D.no_mask(
            values=np.ones((21, 21)), pixel_scales=pixel_scales
        ),
        psf=psf_fine,
        over_sample_size_lp=3,  # not divisible by s=2 (4 became legal with k x s)
        convolve_over_sample_size_lp=2,
    )
except al.exc.DatasetException:
    raised += 1

try:
    al.Imaging(
        data=al.Array2D.no_mask(values=np.zeros((21, 21)), pixel_scales=pixel_scales),
        noise_map=al.Array2D.no_mask(
            values=np.ones((21, 21)), pixel_scales=pixel_scales
        ),
        psf=psf_fine,
        over_sample_size_lp=2,
        convolve_over_sample_size_lp=2.0,
    )
except TypeError:
    raised += 1

assert raised == 3, f"expected 3 guard raises, got {raised}"

print("Guards PASSED (sparse operator, non-divisible over-sample size, non-int size)")

print("\nOversampled PSF convolution tests PASSED")

"""
__Simulate -> Fit Round Trip__

The simulator's oversampled path (evaluate the padded image at the fine resolution, convolve, bin) must be
exactly consistent with the fit's: simulating noise-free at s=2 and fitting the same tracer at s=2 gives
chi_squared ~ 0.
"""
psf_sim = al.Convolver.from_gaussian(
    shape_native=(21, 21),
    pixel_scales=pixel_scales / s,
    sigma=0.15,
    normalize=True,
    convolve_over_sample_size=s,
)

grid_sim = al.Grid2D.uniform(
    shape_native=(21, 21), pixel_scales=pixel_scales, over_sample_size=s
)

simulator = al.SimulatorImaging(
    exposure_time=300.0, psf=psf_sim, add_poisson_noise_to_data=False
)

dataset_sim = simulator.via_tracer_from(tracer=tracer_lp, grid=grid_sim)
dataset_sim.noise_map = al.Array2D.ones(
    shape_native=dataset_sim.data.shape_native, pixel_scales=pixel_scales
)

masked_sim = al.Imaging(
    data=dataset_sim.data,
    noise_map=dataset_sim.noise_map,
    psf=psf_sim,
    over_sample_size_lp=s,
    over_sample_size_pixelization=s,
    convolve_over_sample_size_lp=s,
    convolve_over_sample_size_pixelization=s,
).apply_mask(mask=mask)

fit_sim = al.FitImaging(dataset=masked_sim, tracer=tracer_lp)

assert (
    fit_sim.chi_squared < 1.0e-8
), f"simulate->fit round trip chi_squared = {fit_sim.chi_squared}"

print(f"Simulate -> fit round trip PASSED  (chi2: {fit_sim.chi_squared:.3e})")

print("\nOversampled PSF convolution tests PASSED (including simulator round trip)")

"""
__Adaptive Evaluation (k x s Coupling)__

Adaptive over sampling composes with oversampled convolution: every evaluation size must be divisible by
`convolve_over_sample_size` (k x s), with each pixel's k_i * s evaluation partially binned to the uniform
convolution resolution before blurring. The workspace's adaptive radial schemes therefore work unchanged
with an oversampled PSF.
"""
sizes_adaptive = al.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=al.Grid2D.from_mask(mask=mask, over_sample_size=1),
    sub_size_list=[8, 4, 2],
    radial_list=[0.4, 1.0],
    centre_list=[(0.0, 0.0)],
)

dataset_adaptive = dataset_from(
    tracer=tracer_lp, psf=psf_fine, convolve_over_sample_size=s
).apply_over_sampling(over_sample_size_lp=sizes_adaptive)

# Self-consistency: the data must be rebuilt through the SAME adaptive evaluation,
# then the fit of the same tracer at the same settings gives chi_squared ~ 0.
blurred_adaptive = tracer_lp.blurred_image_2d_from(
    grid=dataset_adaptive.grids.lp,
    blurring_grid=dataset_adaptive.grids.blurring,
    psf=dataset_adaptive.psf,
)

dataset_adaptive = (
    al.Imaging(
        data=al.Array2D(values=np.array(blurred_adaptive), mask=mask).native,
        noise_map=al.Array2D.no_mask(
            values=np.ones((21, 21)), pixel_scales=pixel_scales
        ),
        psf=psf_fine,
        over_sample_size_lp=s,
        over_sample_size_pixelization=s,
        convolve_over_sample_size_lp=s,
        convolve_over_sample_size_pixelization=s,
    )
    .apply_mask(mask=mask)
    .apply_over_sampling(over_sample_size_lp=sizes_adaptive)
)

fit_adaptive = al.FitImaging(dataset=dataset_adaptive, tracer=tracer_lp)

assert (
    fit_adaptive.chi_squared < 1.0e-8
), f"adaptive k x s chi_squared = {fit_adaptive.chi_squared}"

# The adaptive evaluation measurably differs from uniform-s evaluation of the
# same model (the finer central integration is doing real work).
diff = np.max(np.abs(np.array(blurred_adaptive) - np.array(dataset_s2.data)))
assert diff > 1.0e-8, "adaptive evaluation changed nothing — k x s inert?"

print(
    f"Adaptive k x s (lp surfaces) PASSED  (chi2: {fit_adaptive.chi_squared:.3e}, "
    f"max|adaptive - uniform| = {diff:.3e})"
)

"""
Pixelized sources under adaptive pixelization over sampling: the k x s mapping matrix is exact by
linearity, so the fit runs and reconstructs the data as before.
"""
sizes_pix = al.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=al.Grid2D.from_mask(mask=mask, over_sample_size=1),
    sub_size_list=[4, 2],
    radial_list=[0.8],
    centre_list=[(0.0, 0.0)],
)

dataset_pix_adaptive = dataset_pix.apply_over_sampling(
    over_sample_size_pixelization=sizes_pix
)

fit_pix_adaptive = al.FitImaging(dataset=dataset_pix_adaptive, tracer=tracer_pix)

assert np.isfinite(fit_pix_adaptive.log_evidence)
assert (
    fit_pix_adaptive.chi_squared < 1.0e-1
), f"adaptive pixelized k x s chi_squared = {fit_pix_adaptive.chi_squared}"

print(
    f"Adaptive k x s (pixelized, mapping formalism) PASSED  "
    f"(chi2: {fit_pix_adaptive.chi_squared:.3e})"
)

"""
The divisibility guard: evaluation sizes not divisible by the convolution size raise loudly.
"""
try:
    al.Imaging(
        data=al.Array2D.no_mask(values=np.zeros((21, 21)), pixel_scales=pixel_scales),
        noise_map=al.Array2D.no_mask(
            values=np.ones((21, 21)), pixel_scales=pixel_scales
        ),
        psf=psf_fine,
        over_sample_size_lp=3,
        convolve_over_sample_size_lp=2,
    )
    raise AssertionError("divisibility guard did not raise")
except al.exc.DatasetException:
    pass

print("Divisibility guard PASSED")

"""
Simulate -> fit round trip on an adaptive grid: the simulator's padded frame inherits the adaptive
evaluation sizes (border padded with s), so simulation and fitting share the same k x s machinery.
"""
grid_sim_adaptive = al.Grid2D.uniform(
    shape_native=(21, 21), pixel_scales=pixel_scales, over_sample_size=s
)
sizes_sim = al.util.over_sample.over_sample_size_via_radial_bins_from(
    grid=grid_sim_adaptive,
    sub_size_list=[8, 4, 2],
    radial_list=[0.4, 1.0],
    centre_list=[(0.0, 0.0)],
)
grid_sim_adaptive = grid_sim_adaptive.apply_over_sampling(over_sample_size=sizes_sim)

simulator_adaptive = al.SimulatorImaging(
    exposure_time=300.0, psf=psf_sim, add_poisson_noise_to_data=False
)
dataset_sim_adaptive = simulator_adaptive.via_tracer_from(
    tracer=tracer_lp, grid=grid_sim_adaptive
)
dataset_sim_adaptive.noise_map = al.Array2D.ones(
    shape_native=dataset_sim_adaptive.data.shape_native, pixel_scales=pixel_scales
)

masked_sim_adaptive = (
    al.Imaging(
        data=dataset_sim_adaptive.data,
        noise_map=dataset_sim_adaptive.noise_map,
        psf=psf_sim,
        over_sample_size_lp=s,
        over_sample_size_pixelization=s,
        convolve_over_sample_size_lp=s,
        convolve_over_sample_size_pixelization=s,
    )
    .apply_mask(mask=mask)
    .apply_over_sampling(over_sample_size_lp=sizes_adaptive)
)

fit_sim_adaptive = al.FitImaging(dataset=masked_sim_adaptive, tracer=tracer_lp)

assert (
    fit_sim_adaptive.chi_squared < 1.0e-8
), f"adaptive simulate->fit chi_squared = {fit_sim_adaptive.chi_squared}"

print(
    f"Adaptive simulate -> fit round trip PASSED  (chi2: {fit_sim_adaptive.chi_squared:.3e})"
)

print("\nk x s coupling workspace tests PASSED")
