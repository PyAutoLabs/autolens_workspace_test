"""
NUFFT Parity: nufftax (TransformerNUFFT) vs the exact TransformerDFT
=====================================================================

This script verifies that **nufftax** (https://github.com/GragasLab/nufftax),
the backend behind PyAutoLens's ``TransformerNUFFT``, produces numerically
correct visibilities for the interferometer image-to-visibility forward NUFFT
and its adjoint. ``TransformerDFT`` — a direct, unapproximated Fourier
transform — is the reference throughout.

History: this script began as the parity check for swapping pynufft for
nufftax inside ``TransformerNUFFT``, which unblocked end-to-end JAX-jit'd
interferometer likelihoods. That swap has shipped and pynufft has since been
removed from PyAutoArray entirely, so the pynufft legs are gone; what remains
is the standing correctness check of nufftax against the exact DFT.

The script mirrors ``scripts/imaging/convolution.py`` in structure:
auto-simulate dataset, build a tracer image, compute via two implementations,
print residuals, save a residuals image, and assert numerical agreement.

Convention recipe
-----------------
nufftax computes (with ``isign=-1``, ``modeord=0``, default CMCL ordering):

    c[j] = sum_{k1, k2} f[k2, k1] * exp(-i * (k1 * x[j] + k2 * y[j]))

where ``f`` has shape ``(n2, n1)``, ``k1`` ranges over ``-n1//2 .. n1//2-1``,
``k2`` over ``-n2//2 .. n2//2-1``, and ``x, y`` are non-uniform points in
``[-pi, pi)``. The recipe to match autoarray's ``TransformerDFT`` is:

    image_flipped = image[::-1, :]                  # autoarray row 0 = top (y up); nufftax row 0 = mode -n2//2
    x = 2 * pi * u_lambda * pixel_scale_rad         # x is the col-axis (x) frequency
    y = 2 * pi * v_lambda * pixel_scale_rad         # y is the row-axis (y) frequency
    offset_x = 0.5 if N_x is even else 0.0          # parity-dependent grid-centre offset:
    offset_y = 0.5 if N_y is even else 0.0          # autoarray's centre is at (N-1)/2, nufftax's mode 0 is at N//2
    shift = exp(-i * (offset_x * x + offset_y * y))
    visibilities = nufft2d2(x, y, image_flipped, eps=1e-12, isign=-1) * shift

For the typical even-by-even image (e.g. 256x256), ``shift = exp(-0.5j*(x+y))``,
which is the same expression as ``TransformerNUFFT``'s internal ``_shift``.
It is **required** for nufftax, which does not apply the half-pixel correction
internally.

Test cases
----------
(a) All-ones 5x5 image, 0.005" pixels, 3 uv points -- low-noise sanity anchor
    that reproduces the TransformerNUFFT pytest fixture.
(b) Lensed Sersic image, 128x128, real SMA uv coverage (190 visibilities) --
    the realistic case used by the JAX likelihood scripts.
(c) Mapping matrix with 2 columns -- exercises the ``transform_mapping_matrix``
    code path used by source pixelizations.
(d) Adjoint / image_from -- inverse direction (visibilities -> image),
    using nufftax ``nufft2d1``.

Usage
-----
Run from the ``autolens_workspace_test/`` repo root::

    NUMBA_CACHE_DIR=/tmp/numba_cache MPLCONFIGDIR=/tmp/matplotlib \\
        python scripts/interferometer/nufft.py

This script runs at **full resolution**: test (b) is the realistic 128x128 /
0.2" production geometry the JAX likelihood scripts use, and it guards that
geometry explicitly. nufftax's accuracy is not grid-size dependent (measured
3.0e-14 vs 1.7e-14 relative at 16x16 and 256x256), so the tolerances here
would survive a reduced-resolution profile -- the full-resolution requirement
is now about testing the production-scale problem, not about a tolerance that
only holds at one N.

__Env__

Test-harness configuration (PyAutoHands docs/env_profile_redesign.md §10).
Numerical-precision test exercised at the documented production 128x128 / 0.2"
geometry. No search and no plotting fidelity is needed, so only the dataset cap
is released.

ENV: full_datasets
"""

import os

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

from os import path
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp
from astropy import units

import autolens as al
import nufftax

jax.config.update("jax_enable_x64", True)


# =============================================================================
# nufftax helpers
# =============================================================================


def visibilities_via_nufftax(
    image_native_2d: np.ndarray,
    uv_wavelengths: np.ndarray,
    pixel_scales,
    eps: float = 1e-12,
) -> np.ndarray:
    """Forward NUFFT (image -> visibilities) via nufftax, matching
    autoarray's ``TransformerDFT`` / ``TransformerNUFFT`` convention.

    Parameters
    ----------
    image_native_2d
        Real-space image, shape ``(N_y, N_x)``, in autoarray native orientation
        (row 0 = top of image, y increasing upward in physical coords).
    uv_wavelengths
        Visibility (u, v) coordinates in wavelengths, shape ``(M, 2)``.
    pixel_scales
        ``(scale_y, scale_x)`` pixel scales in arcseconds. Only ``scale_y`` is
        used (assumed isotropic, matching the existing ``TransformerNUFFT``).
    eps
        Requested NUFFT precision.

    Returns
    -------
    Visibilities, shape ``(M,)`` complex128.
    """
    pixel_scale_rad = pixel_scales[0] * units.arcsec.to(units.rad)
    img = jnp.asarray(image_native_2d[::-1, :].astype(np.complex128))
    x = jnp.asarray(2.0 * np.pi * uv_wavelengths[:, 0] * pixel_scale_rad)
    y = jnp.asarray(2.0 * np.pi * uv_wavelengths[:, 1] * pixel_scale_rad)
    n_y, n_x = image_native_2d.shape
    offset_x = 0.5 if n_x % 2 == 0 else 0.0
    offset_y = 0.5 if n_y % 2 == 0 else 0.0
    shift = jnp.exp(-1j * (offset_x * x + offset_y * y))
    return np.asarray(nufftax.nufft2d2(x, y, img, eps, -1) * shift)


def image_via_nufftax_adjoint(
    visibilities: np.ndarray,
    uv_wavelengths: np.ndarray,
    pixel_scales,
    shape_native,
    eps: float = 1e-12,
) -> np.ndarray:
    """Adjoint NUFFT (visibilities -> image) via nufftax ``nufft2d1``.

    The adjoint inverts the half-pixel shift applied in the forward path
    by multiplying ``visibilities`` by ``conj(shift)`` before the type-1 call,
    then unflips the row axis to return to autoarray native orientation.

    Returns the real part of the adjoint (matching ``TransformerNUFFT.image_from``,
    which also discards imaginary residue).
    """
    pixel_scale_rad = pixel_scales[0] * units.arcsec.to(units.rad)
    x = jnp.asarray(2.0 * np.pi * uv_wavelengths[:, 0] * pixel_scale_rad)
    y = jnp.asarray(2.0 * np.pi * uv_wavelengths[:, 1] * pixel_scale_rad)
    n_y, n_x = shape_native
    offset_x = 0.5 if n_x % 2 == 0 else 0.0
    offset_y = 0.5 if n_y % 2 == 0 else 0.0
    shift = jnp.exp(-1j * (offset_x * x + offset_y * y))
    c = jnp.asarray(visibilities) * jnp.conj(shift)
    n_modes = (n_x, n_y)  # (n1, n2) = (N_x, N_y)
    f = nufftax.nufft2d1(x, y, c, n_modes, eps, +1)
    return np.asarray(f)[::-1, :].real


def transform_mapping_matrix_via_nufftax(
    mapping_matrix: np.ndarray,
    mask: al.Mask2D,
    uv_wavelengths: np.ndarray,
    pixel_scales,
    eps: float = 1e-12,
) -> np.ndarray:
    """Apply the forward NUFFT to each column of a mapping matrix
    (mirrors ``TransformerNUFFT.transform_mapping_matrix``).
    """
    n_uv = uv_wavelengths.shape[0]
    n_src = mapping_matrix.shape[1]
    out = np.zeros((n_uv, n_src), dtype=np.complex128)
    for k in range(n_src):
        image_2d = np.zeros(mask.shape, dtype=np.float64)
        image_2d[mask.slim_to_native_tuple] = mapping_matrix[:, k]
        out[:, k] = visibilities_via_nufftax(
            image_2d, uv_wavelengths, pixel_scales, eps=eps
        )
    return out


# =============================================================================
# Test (a): all-ones 5x5 image -- replicates the TransformerNUFFT pytest fixture
# =============================================================================

print("=" * 70)
print("(a) All-ones 5x5 image: low-noise convention check")
print("=" * 70)

uv_a = np.array([[0.2, 1.0], [0.5, 1.1], [0.8, 1.2]], dtype=np.float64)
mask_a = al.Mask2D.all_false(shape_native=(5, 5), pixel_scales=0.005)
image_a = al.Array2D.ones(shape_native=(5, 5), pixel_scales=0.005)

dft_a = al.TransformerDFT(uv_wavelengths=uv_a, real_space_mask=mask_a)
# `al.TransformerNUFFT` is the shipped nufftax-backed transformer. It is
# compared against BOTH the exact DFT and the local nufftax recipe below, so
# the check does not degenerate into comparing the library against itself.
lib_a = al.TransformerNUFFT(uv_wavelengths=uv_a, real_space_mask=mask_a)

vis_a_dft = np.asarray(dft_a.visibilities_from(image=image_a))
vis_a_nfx = visibilities_via_nufftax(image_a.native.array, uv_a, mask_a.pixel_scales)
vis_a_lib = np.asarray(lib_a.visibilities_from(image=image_a.native))

print(f"vis (DFT)     : {vis_a_dft}")
print(f"vis (nufftax) : {vis_a_nfx}")
print(f"vis (library) : {vis_a_lib}")
print(f"max |Δ| nufftax - DFT     : {np.max(np.abs(vis_a_nfx - vis_a_dft)):.4e}")
print(f"max |Δ| library - DFT     : {np.max(np.abs(vis_a_lib - vis_a_dft)):.4e}")

# nufftax matches the analytic DFT to machine precision.
assert (
    np.max(np.abs(vis_a_nfx - vis_a_dft)) < 1e-10
), "nufftax should match TransformerDFT exactly on all-ones 5x5"
# The shipped transformer must match the same exact reference.
assert (
    np.max(np.abs(vis_a_lib - vis_a_dft)) < 1e-10
), "al.TransformerNUFFT should match TransformerDFT exactly on all-ones 5x5"
# The shipped `al.TransformerNUFFT` must reproduce the local nufftax recipe.
assert (
    np.max(np.abs(vis_a_lib - vis_a_nfx)) < 1e-10
), "al.TransformerNUFFT should match the local nufftax recipe on all-ones 5x5"


# =============================================================================
# Test (b): Lensed Sersic image, 128x128, real SMA uv (the production case)
# =============================================================================

print()
print("=" * 70)
print("(b) Lensed Sersic image, 128x128, SMA uv coverage")
print("=" * 70)

dataset_path = path.join("dataset", "interferometer", "simple")

real_space_mask = al.Mask2D.circular(
    shape_native=(128, 128),
    pixel_scales=0.2,
    radius=3.0,
)

# Guard the production geometry this test is meant to exercise. `Mask2D.circular`
# silently honours `PYAUTO_SMALL_DATASETS=1` by capping to (16, 16) at 0.6".
# Without this guard the script keeps printing "128x128" while comparing a 16x16
# problem — a silently weaker test rather than the geometry error it is. (When
# this script still had a gridding-based backend, that mismatch surfaced ~60
# lines later as a blown tolerance; nufftax is accurate at both sizes, so the
# guard is now the only thing that catches it.) The `ENV: full_datasets` declaration
# at the top of this file is what keeps this assertion true under the smoke and
# release profiles.
#
# It runs BEFORE `should_simulate` deliberately: that call deletes the on-disk
# dataset when the cap is active, so guarding first means a misconfigured run
# fails without first destroying a full-resolution dataset other scripts share.
assert real_space_mask.shape_native == (128, 128) and real_space_mask.pixel_scales == (
    0.2,
    0.2,
), (
    f'Test (b) requires the full-resolution 128x128 / 0.2" grid, but got '
    f"{real_space_mask.shape_native} at {real_space_mask.pixel_scales}. "
    f"PYAUTO_SMALL_DATASETS is capping it; this script declares "
    f"`ENV: full_datasets` to release that cap."
)

if al.util.dataset.should_simulate(dataset_path):
    print("Dataset missing - running simulator...")
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

print(f"Total visibilities: {dataset.uv_wavelengths.shape[0]}")

lens_galaxy = al.Galaxy(
    redshift=0.5,
    mass=al.mp.Isothermal(
        centre=(0.0, 0.0),
        einstein_radius=1.6,
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
    ),
)

# External Shear: the tidal field of everything outside the modelled system, so it is a property of the
# system rather than of the lens galaxy. It is held in an `al.MassField` — a container like a galaxy
# which carries no light — and passed to the tracer's `fields=` argument. The tracer sums every
# deflection field over the plane, so the simulated data is unchanged.
field = al.MassField(
    redshift=0.5, shear=al.mp.ExternalShear(gamma_1=0.05, gamma_2=0.05)
)
source_galaxy = al.Galaxy(
    redshift=1.0,
    bulge=al.lp.SersicCore(
        centre=(0.1, 0.1),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=60.0),
        intensity=0.3,
        effective_radius=1.0,
        sersic_index=2.5,
    ),
)
tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy], fields=[field])

# image_2d_from returns the masked slim representation; .native gives the 2D array
# with masked pixels set to zero, which is what both transformers expect.
image_b = tracer.image_2d_from(grid=dataset.grid)
image_b_native = image_b.native.array

dft_b = al.TransformerDFT(
    uv_wavelengths=dataset.uv_wavelengths, real_space_mask=real_space_mask
)
lib_b = al.TransformerNUFFT(
    uv_wavelengths=dataset.uv_wavelengths, real_space_mask=real_space_mask
)

print("Running TransformerDFT (slow, exact reference)...")
vis_b_dft = np.asarray(dft_b.visibilities_from(image=image_b))

vis_b_nfx = visibilities_via_nufftax(
    image_b_native, dataset.uv_wavelengths, real_space_mask.pixel_scales
)
vis_b_lib = np.asarray(lib_b.visibilities_from(image=image_b.native))

dft_scale = float(np.max(np.abs(vis_b_dft)))
print(f"|vis_DFT|_max = {dft_scale:.4e}")
print(
    f"max |Δ| nufftax - DFT     : "
    f"{np.max(np.abs(vis_b_nfx - vis_b_dft)):.4e}  "
    f"(rel: {np.max(np.abs(vis_b_nfx - vis_b_dft)) / dft_scale:.4e})"
)
print(
    f"max |Δ| library - DFT     : "
    f"{np.max(np.abs(vis_b_lib - vis_b_dft)):.4e}  "
    f"(rel: {np.max(np.abs(vis_b_lib - vis_b_dft)) / dft_scale:.4e})"
)

# nufftax with eps=1e-12 is effectively exact; match DFT to ~1e-9 relative.
assert (
    np.max(np.abs(vis_b_nfx - vis_b_dft)) / dft_scale < 1e-9
), "nufftax should match TransformerDFT to ~1e-9 relative on 128x128"
# The shipped transformer must match the same exact reference. Unlike the
# gridding-based backend this script was originally written against, nufftax's
# accuracy does not degrade at small N (measured 2026-08-04: 3.0e-14 relative
# at 256x256 / 0.1", 1.7e-14 at 16x16 / 0.6"), so this tolerance is not a
# resolution-specific number.
assert (
    np.max(np.abs(vis_b_lib - vis_b_dft)) / dft_scale < 1e-9
), "al.TransformerNUFFT should match TransformerDFT to ~1e-9 relative on 128x128"
# The shipped `al.TransformerNUFFT` must reproduce the local nufftax recipe.
assert (
    np.max(np.abs(vis_b_lib - vis_b_nfx)) / dft_scale < 1e-9
), "al.TransformerNUFFT should match the local nufftax recipe on 128x128"


# Save residuals plot for visual sanity check (mirrors imaging/convolution.py)
script_path = Path("scripts") / "interferometer" / "images"
script_path.mkdir(parents=True, exist_ok=True)

# Two panels, not three: the shipped transformer and the local nufftax recipe
# agree bit-for-bit (exactly 0.0 residual), so a third panel plotting their
# difference is an empty log-scale axis rather than information. The equality
# is still asserted below.
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(np.abs(vis_b_dft), "k-", label="|DFT|", lw=0.8)
axes[0].set_title("|visibilities| (DFT reference)")
axes[0].set_xlabel("uv index")
axes[1].plot(np.abs(vis_b_nfx - vis_b_dft), "b-", label="nufftax - DFT", lw=0.8)
axes[1].plot(np.abs(vis_b_lib - vis_b_dft), "r--", label="library - DFT", lw=0.8)
axes[1].set_yscale("log")
axes[1].set_title("|residual vs DFT|")
axes[1].set_xlabel("uv index")
axes[1].legend()
plt.tight_layout()
plt.savefig(script_path / "nufft_residuals.png", dpi=150)
plt.close(fig)
print(f"Saved residuals plot to {script_path / 'nufft_residuals.png'}")


# =============================================================================
# Test (c): Mapping matrix with 2 columns
# =============================================================================

print()
print("=" * 70)
print("(c) Mapping matrix transform")
print("=" * 70)

# Two source-pixel basis functions in the masked slim representation
mapping_matrix = np.zeros((image_b.shape[0], 2), dtype=np.float64)
mapping_matrix[:, 0] = image_b.array
mapping_matrix[:, 1] = image_b.array * 0.5 + 0.1  # second column is different

mm_dft = np.asarray(dft_b.transform_mapping_matrix(mapping_matrix=mapping_matrix))
mm_nfx = transform_mapping_matrix_via_nufftax(
    mapping_matrix=mapping_matrix,
    mask=real_space_mask,
    uv_wavelengths=dataset.uv_wavelengths,
    pixel_scales=real_space_mask.pixel_scales,
)

mm_scale = float(np.max(np.abs(mm_dft)))
print(f"mapping matrix shape: {mm_dft.shape}")
print(f"|mm_DFT|_max = {mm_scale:.4e}")
print(
    f"max |Δ| nufftax - DFT     : "
    f"{np.max(np.abs(mm_nfx - mm_dft)):.4e}  "
    f"(rel: {np.max(np.abs(mm_nfx - mm_dft)) / mm_scale:.4e})"
)

# The exact DFT is the reference here. This tolerance is far tighter than the
# gridding-precision one it replaces, because nufftax is essentially exact.
assert np.max(np.abs(mm_nfx - mm_dft)) / mm_scale < 1e-9, (
    "nufftax mapping matrix must agree with the exact TransformerDFT " "mapping matrix"
)

# The shipped `al.TransformerNUFFT.transform_mapping_matrix` (batched, one
# nufft2d2 call) must reproduce the per-column local nufftax recipe.
mm_lib = np.asarray(lib_b.transform_mapping_matrix(mapping_matrix=mapping_matrix))
print(
    f"max |Δ| library - nufftax : "
    f"{np.max(np.abs(mm_lib - mm_nfx)):.4e}  "
    f"(rel: {np.max(np.abs(mm_lib - mm_nfx)) / mm_scale:.4e})"
)
assert np.max(np.abs(mm_lib - mm_nfx)) / mm_scale < 1e-9, (
    "al.TransformerNUFFT.transform_mapping_matrix must match the per-column "
    "nufftax recipe"
)


# =============================================================================
# Test (d): Adjoint -- image_from
# =============================================================================
#
# The adjoint is checked by property rather than against a second
# implementation: adjoint normalisation is library-specific, so a bit-for-bit
# comparison of raw ``image_from`` output across backends is not meaningful.
#
# Instead we verify two well-defined properties that any correct adjoint
# pair must satisfy:
#
#   (d.1) Library-internal adjoint identity for nufftax:
#         <nufft2d1(c), f> = <c, nufft2d2(f)>*  (within numerical precision)
#         If this fails, nufftax's two transforms are not a true adjoint pair
#         and gradient computation through them would be wrong.
#
#   (d.2) Forward -> adjoint round trip puts brightness where the image was:
#         Apply nufftax forward to a known lensed image to get visibilities,
#         then apply nufftax adjoint to those visibilities. The resulting
#         "dirty image" should peak near the brightest pixel of the original.
#         This proves the adjoint is correctly oriented relative to the
#         forward (no row/column sign flip).

print()
print("=" * 70)
print("(d) Adjoint NUFFT (visibilities -> image)")
print("=" * 70)

pixel_scale_rad = real_space_mask.pixel_scales[0] * units.arcsec.to(units.rad)
x_jx = jnp.asarray(2.0 * np.pi * dataset.uv_wavelengths[:, 0] * pixel_scale_rad)
y_jx = jnp.asarray(2.0 * np.pi * dataset.uv_wavelengths[:, 1] * pixel_scale_rad)
n_modes_probe = (
    real_space_mask.shape_native[1],
    real_space_mask.shape_native[0],
)

# (d.1) Adjoint identity for nufftax
rng = np.random.default_rng(0)
c_probe = rng.standard_normal(
    dataset.uv_wavelengths.shape[0]
) + 1j * rng.standard_normal(dataset.uv_wavelengths.shape[0])
f_probe = rng.standard_normal(real_space_mask.shape_native) + 1j * rng.standard_normal(
    real_space_mask.shape_native
)
img_from_c = np.asarray(
    nufftax.nufft2d1(x_jx, y_jx, jnp.asarray(c_probe), n_modes_probe, 1e-12, +1)
)
vis_from_f = np.asarray(nufftax.nufft2d2(x_jx, y_jx, jnp.asarray(f_probe), 1e-12, -1))
# Standard adjoint identity (derived in nufftax/transforms/autodiff.py: Type 1
# and Type 2 are adjoints of each other with opposite isign):
#   sum_k nufft2d1(c)[k] * conj(f[k]) == sum_j c[j] * conj(nufft2d2(f)[j])
inner_lhs = np.sum(img_from_c * np.conj(f_probe))
inner_rhs = np.sum(c_probe * np.conj(vis_from_f))
adjoint_residual = abs(inner_lhs - inner_rhs) / max(abs(inner_lhs), abs(inner_rhs), 1.0)
print(
    f"(d.1) nufftax adjoint identity "
    f"|<nufft2d1(c), f> - <c, nufft2d2(f)>|_rel : "
    f"{adjoint_residual:.4e}"
)
assert (
    adjoint_residual < 1e-9
), "nufftax must satisfy the adjoint property between nufft2d1 and nufft2d2"

# (d.2) Forward -> adjoint round trip on a known image
# We use the lensed image from test (b): push it through the nufftax forward
# transform, then through the nufftax adjoint, and check the resulting dirty
# image two ways — against the exact DFT adjoint of the same visibilities
# (d.2a) and by how far the flux-weighted centroid moved (d.2b).
vis_round = visibilities_via_nufftax(
    image_b_native, dataset.uv_wavelengths, real_space_mask.pixel_scales
)
img_round = image_via_nufftax_adjoint(
    vis_round,
    dataset.uv_wavelengths,
    real_space_mask.pixel_scales,
    shape_native=real_space_mask.shape_native,
)

# (d.2a) The adjoint must reproduce the exact DFT adjoint of the same
# visibilities. `dft_b` (built in test (b)) carries the same uv coverage and
# real-space mask, so `image_from` is the unapproximated reference for
# `image_via_nufftax_adjoint`, exactly as `visibilities_from` was for the
# forward direction. This is the assertion that actually tests nufftax, and
# like (a)/(b)/(c) it is a relative-accuracy number, not a positional slop.
img_round_dft = np.asarray(
    dft_b.image_from(
        visibilities=al.Visibilities(
            np.stack([vis_round.real, vis_round.imag], axis=-1)
        )
    ).native.array
)
unmasked = ~np.asarray(real_space_mask)
dirty_scale = float(np.max(np.abs(img_round_dft[unmasked])))
adjoint_rel = float(
    np.max(np.abs(img_round[unmasked] - img_round_dft[unmasked])) / dirty_scale
)
print(
    f"(d.2a) round-trip dirty image, nufftax adjoint vs exact DFT adjoint: "
    f"max |Δ| rel = {adjoint_rel:.4e}"
)
# Measured 2026-09-07: 9.4e-15 to 1.8e-14 across 0.05"-0.4" pixel scales
# (64x64 through 512x512), i.e. machine precision and grid-size independent —
# the same 1e-9 relative tolerance the forward tests use.
assert (
    adjoint_rel < 1e-9
), f"nufftax adjoint should match the exact DFT adjoint to ~1e-9 relative: {adjoint_rel:.4e}"

# (d.2b) The round trip must not translate the image. The right positional
# statement is the shift of the flux-weighted centroid inside the real-space
# mask, in arcseconds — NOT the pixel distance between the two argmax pixels,
# which this check used until 2026-09-07.
#
# Why the argmax was wrong: this is a two-image lens, and the dirty beam
# (190 SMA baselines) smooths the extended counter-image more favourably than
# the compact arc. Original and dirty image therefore each have two nearly
# equal maxima, and which one wins is a tie-break, not a property of the
# transform. Measured 2026-09-07 on the shipped 128x128/0.2" data: the dirty
# image at the original's peak is 96.1% of its own peak. Coarsening the grid
# from 256x256/0.1" to 128x128/0.2" in `fb6e709` flipped the original's argmax
# to the other lensed image, and the "distance" jumped 3.61 px (0.361") to
# 15.81 px (3.162") — the image separation — with the transform unchanged
# (d.2a is 1e-14 in both geometries, and the numbers are identical on the
# pre-2026-08-31 library). Any pixel or angular pin on that distance is a pin
# on a coin flip, so it is replaced rather than rescaled.
peak_image = np.unravel_index(np.argmax(np.abs(image_b_native)), image_b_native.shape)
peak_round = np.unravel_index(np.argmax(np.abs(img_round)), img_round.shape)


def centroid_2d(array_2d, mask_2d):
    """Flux-weighted (row, col) centroid of ``|array_2d|`` over the unmasked pixels."""
    weights = np.abs(array_2d) * mask_2d
    rows, cols = np.indices(weights.shape)
    total = weights.sum()
    return (weights * rows).sum() / total, (weights * cols).sum() / total


centroid_image = centroid_2d(image_b_native, unmasked)
centroid_round = centroid_2d(img_round, unmasked)
centroid_shift_pixels = float(
    np.sqrt(
        (centroid_image[0] - centroid_round[0]) ** 2
        + (centroid_image[1] - centroid_round[1]) ** 2
    )
)
centroid_shift_arcsec = centroid_shift_pixels * real_space_mask.pixel_scales[0]
print(
    f"(d.2b) round trip: peak of original = {peak_image}, "
    f"peak of dirty image = {peak_round}  (argmax, not asserted — see comment); "
    f"centroid shift = {centroid_shift_pixels:.3f} px "
    f'= {centroid_shift_arcsec:.4f}"'
)
# Unlike the argmax distance, the centroid shift is a geometry-independent
# angular quantity: measured 2026-09-07 as 0.0359", 0.0362", 0.0364", 0.0366",
# 0.0450" and 0.0506" at pixel scales 0.05", 0.15", 0.1", 0.2", 0.3" and 0.4"
# (an 8x range), i.e. always a small fraction of a beam and stable to 0.005".
# The 0.06" pin is the worst of those (0.0506", the coarsest grid tested) with
# the same ~1.2x margin the old 5.0 -> 6.0 pixel bump used; on the shipped 0.2"
# geometry the margin is 1.6x. Verified non-vacuous 2026-09-07: it fires on a
# one-pixel translation of the adjoint output, and (d.2a) additionally fires on
# a dropped half-pixel shift and on a 5% gain error — three hazards the retired
# `distance < 6.0` assert passed unchanged.
assert centroid_shift_arcsec < 0.06, (
    f"Round-trip dirty image is displaced from the original: "
    f'centroid moved {centroid_shift_arcsec:.4f}" '
    f"({centroid_shift_pixels:.3f} px)"
)


print()
print("=" * 70)
print("All NUFFT parity tests PASSED.")
print("=" * 70)
print(
    "Convention recipe (image -> visibilities):\n"
    "  image_flipped = image[::-1, :]\n"
    "  x = 2*pi * u_lambda * pixel_scale_rad\n"
    "  y = 2*pi * v_lambda * pixel_scale_rad\n"
    "  offset_x = 0.5 if N_x % 2 == 0 else 0.0\n"
    "  offset_y = 0.5 if N_y % 2 == 0 else 0.0\n"
    "  shift = exp(-1j * (offset_x * x + offset_y * y))\n"
    "  visibilities = nufft2d2(x, y, image_flipped, eps, -1) * shift\n"
)
