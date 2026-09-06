"""
Over Sampling: Adapt From A Signal-To-Noise Map
==============================================

Regression guard for the SLaM adapt-image over-sampling arithmetic (autolens_workspace#523).

The source adapt image used by the SLaM pipelines is
`al.galaxy_name_image_dict_via_result_from(result)[...]`, which is already a
**signal-to-noise map** (`subtracted_image / noise_map`). Five workspace scripts passed it as
`data=` to `al.util.over_sample.over_sample_size_via_adapt_from(data, noise_map, ...)`, whose
first operation is `data / noise_map` — a second division by the noise. The signal-to-noise
values it then thresholds are inflated by `1 / noise_map`, so nearly every pixel lands on the
upper sub-size instead of the ~30% that genuinely exceed the cut.

The workspace now thresholds the S/N map directly:

    over_sample_size_pixelization = al.Array2D(
        values=np.where(source_snr > 3.0, 4, 2), mask=dataset.mask
    )

This script pins that idiom against the legacy call on a real dataset:

 1. The direct idiom is exact: the fraction at sub-size 4 equals `mean(snr > cut)` and the map
    holds only {2, 4}.
 2. The legacy call differs (and inflates): its fraction at sub-size 4 is strictly larger.
 3. The resulting map is accepted by `Imaging.apply_over_sampling`, and the pixelization grid
    carries only {2, 4}.

No model fit is run — this is pure over-sampling arithmetic.
"""

# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

from os import path

import numpy as np

import autolens as al

"""
__Dataset__

Load the `with_lens_light` imaging dataset, simulating it first if it is not on disk (or was
written under a different resolution regime).
"""
dataset_label = "build"
dataset_type = "imaging"
dataset_name = "with_lens_light"

dataset_path = path.join("dataset", dataset_label, dataset_type, dataset_name)

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
    pixel_scales=0.3,
)

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=3.0,
)

dataset = dataset.apply_mask(mask=mask)

"""
__Signal-To-Noise Map__

`dataset.signal_to_noise_map` stands in for the per-galaxy adapt image a SLaM result returns via
`galaxy_name_image_dict_via_result_from` — same shape, same mask, same units (S/N, already
divided by the noise-map once).
"""
signal_to_noise_cut = 3.0

snr = np.array(dataset.signal_to_noise_map)

"""
__Workspace Idiom__

Threshold the S/N map directly. The fraction of pixels at the upper sub-size must equal the
fraction of pixels above the cut exactly — no second division, no auto-lowering of the cut.
"""
over_sample = al.Array2D(
    values=np.where(snr > signal_to_noise_cut, 4, 2), mask=dataset.mask
)

frac_4 = np.mean(np.array(over_sample) == 4)
frac_above_cut = np.mean(snr > signal_to_noise_cut)

assert frac_4 == frac_above_cut, (
    f"direct idiom is not an exact threshold of the S/N map: "
    f"frac(sub_size == 4) = {frac_4}, frac(snr > {signal_to_noise_cut}) = {frac_above_cut}"
)

assert set(np.unique(np.array(over_sample))) <= {
    2,
    4,
}, f"over sample map holds values other than {{2, 4}}: {np.unique(np.array(over_sample))}"

"""
__Legacy Call__

`over_sample_size_via_adapt_from` divides `data` by `noise_map` itself, so passing the S/N map as
`data` divides by the noise a second time. The inflated S/N pushes far more pixels above the cut.

The assertions are on the *difference*, not on a pinned fraction: strictly greater (the direction
the double division must push it) and not equal (so the guard still fires if the direction ever
flips, e.g. via the function's auto-lowering of the cut when `max(S/N) < 2 * cut`).
"""
legacy = al.util.over_sample.over_sample_size_via_adapt_from(
    data=dataset.signal_to_noise_map,
    noise_map=dataset.noise_map,
    signal_to_noise_cut=signal_to_noise_cut,
)

frac_4_legacy = np.mean(np.array(legacy) == 4)

assert frac_4_legacy != frac_4, (
    f"legacy double-division call gave the same sub-size 4 fraction as the direct idiom "
    f"({frac_4_legacy}) — the guard would not detect a regression to it"
)

assert frac_4_legacy > frac_4, (
    f"expected the double division to inflate the S/N and raise the sub-size 4 fraction: "
    f"legacy = {frac_4_legacy}, direct = {frac_4}"
)

"""
__Dataset Acceptance__

The direct map must be a valid `over_sample_size_pixelization`: the pixelization grid it produces
carries only the two sub-sizes it was built from.
"""
dataset = dataset.apply_over_sampling(over_sample_size_pixelization=over_sample)

sub_sizes = np.unique(np.array(dataset.grids.pixelization.over_sample_size))

assert set(sub_sizes) <= {
    2,
    4,
}, f"pixelization grid over sample sizes are not {{2, 4}}: {sub_sizes}"

print(f"Direct S/N threshold : frac(sub_size == 4) = {frac_4:.4f}")
print(f"Legacy adapt call    : frac(sub_size == 4) = {frac_4_legacy:.4f}")
