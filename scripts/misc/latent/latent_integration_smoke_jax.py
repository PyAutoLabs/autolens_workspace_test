"""
Integration smoke: a REAL JAX search with a model assertion still writes a full
latent block.
======================================================================

The Euclid ``vis_lp`` stage wrote no latent block at all for months. The model
carried a real ``model.add_assertion(...)`` (the ordered two-basis MGE lens
light), and the latent engine's per-sample ``jax.jit`` applied a Python ``not``
to a traced boolean when it re-checked that assertion. The resulting
``TracerBoolConversionError`` was swallowed into a NaN row for EVERY sample, so
the whole latent block was dropped — silently, with a `files/latent/` directory
that simply never appeared (PyAutoLens#734, PyAutoFit#1600; the fix is
``latent_instance_from(..., ignore_assertions=True)``).

Nothing caught it, because no test ran a JAX search whose model carried an
assertion and then looked at what landed on disk. This script does exactly that:
the lens light is two Gaussians with an explicit ``sigma`` ordering assertion —
the same shape as the ``order_bases`` MGE constraint — fitted through
``al.AnalysisImaging(use_jax=True)`` by a real ``af.Nautilus``.

The complement of ``latent_integration_smoke.py`` (the NumPy leg), which covers
the aggregator and ``af.AggregateCSV`` catalogue path. This one is deliberately
narrow: does the JAX write path produce a complete, finite latent block?

__Speed__

~20 s locally, and unlike the NumPy leg it is XLA-compile bound rather than
work bound: of that, ~5 s builds the JAX likelihood/preloads before sampling,
~1 s compiles the vmapped likelihood, ~7-9 s compiles the PER-SAMPLE latent
function, and only ~0.3 s actually evaluates the four draws. Those compiles are
the code path this script exists to guard, so they cannot be traded away —
cutting draws from 6 to 4 changed the latent pass by under a second, and
shrinking the image or simplifying the source profile changed nothing
measurable. What was cut instead: the image (40x40/11x11 PSF -> 30x30/3x3,
`over_sample_size=1`), the search budget (`n_like_max` 100 -> 40, `n_live` 25 ->
10, `n_networks` 4 -> 1), the PDF draws (6 -> 4) and all visualization.

__Env__ (Developer Only)

``real_search`` is load-bearing: ``skip_latents()`` is true for ANY
``PYAUTO_TEST_MODE`` level, so under the smoke profile's default of ``2`` no
latent block is written and every assertion below would pass on an empty tree.

``jax`` is the point of the script — the defect lives only in the JAX per-sample
trace, and the NumPy path is already covered by the sibling script.

``full_datasets`` is load-bearing: under ``PYAUTO_SMALL_DATASETS=1`` grids and
masks are capped to 15x15, too coarse for the tangential critical curve, so
``effective_einstein_radius`` would come back NaN and be DROPPED from the
summary — the exact-key-set assertion would then fail for the wrong reason.

ENV: real_search full_datasets jax
"""

import json
import math
import os
import shutil
import time
from pathlib import Path

_START = time.perf_counter()
_PHASE_MARK = _START
PHASES = []


def phase(label):
    """
    Record and print the wall time since the previous phase mark, so a drift
    over the 300 s CI cap can be attributed to the fit, the per-sample latent
    jit or the assertions without a bisect.
    """
    global _PHASE_MARK
    now = time.perf_counter()
    PHASES.append((label, now - _PHASE_MARK))
    print(f"[phase] {label}: {now - _PHASE_MARK:.1f}s", flush=True)
    _PHASE_MARK = now


from autonerves.test_mode import skip_latents  # noqa: E402

WORKSPACE = Path(__file__).resolve().parents[3]

"""
__Guards__
"""
assert not skip_latents(), (
    "skip_latents() is True — PYAUTO_TEST_MODE (or PYAUTO_SKIP_LATENTS) is still "
    "set, so the search will write no latent block and every assertion below "
    "would pass vacuously. The `real_search` ENV token must unset it."
)
assert os.environ.get("PYAUTO_SMALL_DATASETS") != "1", (
    "PYAUTO_SMALL_DATASETS=1 caps grids/masks to 15x15, which cannot resolve the "
    "tangential critical curve; the `full_datasets` ENV token must unset it."
)
assert os.environ.get("PYAUTO_DISABLE_JAX") != "1", (
    "PYAUTO_DISABLE_JAX=1 forces use_jax=False; the `jax` ENV token must unset "
    "it, or this script tests the NumPy path the sibling script already covers."
)

"""
__Config Overlay__

A copy of the workspace `config/` with the latent PDF draw count reduced to 4
and every visualization toggle turned off. This leg is about whether a latent
block is written AT ALL under JAX, not about its error bars, so four draws is
enough — and it is still a real per-sample `jax.jit` pass, which is the code
path the guarded regression lives in. Note the cost here is the per-sample JIT
COMPILE, not the draws: the pass took 6.5 s for 6 draws, so cutting draws alone
buys almost nothing and the saving has to come from a smaller graph.
`remove_files` is deliberately left at the workspace default (`false`) so
`search.log` and the latent block are read from the loose output tree; the zip
path is covered by the NumPy leg.
"""
OUTPUT_PATH = WORKSPACE / "output" / "latent_integration_smoke_jax"
CONFIG_PATH = WORKSPACE / "output" / "latent_integration_smoke_jax_config"

for stale in (OUTPUT_PATH, CONFIG_PATH):
    if stale.exists():
        shutil.rmtree(stale)

shutil.copytree(WORKSPACE / "config", CONFIG_PATH)

output_yaml = CONFIG_PATH / "output.yaml"
output_text = output_yaml.read_text()
assert "latent_draw_via_pdf_size : 100" in output_text, (
    "config/output.yaml no longer carries `latent_draw_via_pdf_size : 100`; the "
    "overlay edit below is a no-op."
)
assert "search_log: true" in output_text, (
    "config/output.yaml no longer enables `search_log`; the search.log assertion "
    "at the end of this script would read a file that is never written."
)
output_yaml.write_text(
    output_text.replace(
        "latent_draw_via_pdf_size : 100", "latent_draw_via_pdf_size : 4"
    ).replace("start_point: true", "start_point: false")
)

# Visualization is not read by any assertion here and cost ~2 s. Every
# `plots.yaml` toggle is flipped off in the overlay copy; the repository's own
# config is untouched.
plots_yaml = CONFIG_PATH / "visualize" / "plots.yaml"
plots_yaml.write_text(plots_yaml.read_text().replace(": true", ": false"))

from autolens import conf  # noqa: E402

conf.instance.push(new_path=CONFIG_PATH, output_path=OUTPUT_PATH)

import autofit as af  # noqa: E402
import autolens as al  # noqa: E402
from autolens.analysis.latent import LATENT_FUNCTIONS  # noqa: E402

phase("imports + config overlay")

EXPECTED_KEYS = set(LATENT_FUNCTIONS)
enabled = conf.instance["latent"]
assert EXPECTED_KEYS == {
    key for key in LATENT_FUNCTIONS if enabled[key]
}, "config/latent.yaml no longer enables every entry in LATENT_FUNCTIONS."

MAGZERO = 25.0

"""
__Simulate__

The same 40x40 0.1"/pixel SIE + shear lens and Sersic source as the NumPy leg.
"""
# Sized for SPEED, with one hard floor: the masked light-profile grid must
# still resolve the tangential critical curve, or `einstein_radius_from` returns
# NaN and `effective_einstein_radius` is dropped from the summary. theta_E is
# 1.0", so the 1.3" mask below clears it with room to spare. 30x30 at 0.1" and a
# 3x3 PSF (sigma = 1 pixel) cut the convolution cost ~20x against the 40x40 /
# 11x11 this script started with, and `over_sample_size=1` removes the default
# 4x4 sub-gridding — none of which changes what any assertion below tests.
grid = al.Grid2D.uniform(shape_native=(30, 30), pixel_scales=0.1, over_sample_size=1)

psf = al.Convolver.from_gaussian(
    shape_native=(3, 3),
    sigma=0.1,
    pixel_scales=grid.pixel_scales,
    convolve_over_sample_size=1,
)

LENS_MASS = dict(centre=(0.0, 0.0), ell_comps=(0.05, 0.05), einstein_radius=1.0)
LENS_SHEAR = dict(gamma_1=0.02, gamma_2=0.02)
SOURCE_BULGE = dict(
    centre=(0.0, 0.0),
    ell_comps=(0.0, 0.0),
    intensity=2.0,
    effective_radius=0.2,
    sersic_index=1.0,
)

# The lens light is two concentric Gaussians of different width — the shape the
# `order_bases` MGE constraint has, and the shape the model assertion below
# orders.
GAUSSIAN_INNER = dict(centre=(0.0, 0.0), ell_comps=(0.0, 0.0), intensity=1.0, sigma=0.3)
GAUSSIAN_OUTER = dict(centre=(0.0, 0.0), ell_comps=(0.0, 0.0), intensity=0.5, sigma=0.8)

truth_lens = al.Galaxy(
    redshift=0.5,
    bulge=al.lp.Gaussian(**GAUSSIAN_INNER),
    disk=al.lp.Gaussian(**GAUSSIAN_OUTER),
    mass=al.mp.Isothermal(**LENS_MASS),
    shear=al.mp.ExternalShear(**LENS_SHEAR),
)
truth_source = al.Galaxy(redshift=1.0, bulge=al.lp.Sersic(**SOURCE_BULGE))

simulator = al.SimulatorImaging(
    exposure_time=300.0,
    psf=psf,
    background_sky_level=0.1,
    add_poisson_noise_to_data=True,
    noise_seed=1,
)

dataset = simulator.via_tracer_from(
    tracer=al.Tracer(galaxies=[truth_lens, truth_source]), grid=grid
)
dataset = dataset.apply_mask(
    mask=al.Mask2D.circular(
        shape_native=dataset.shape_native,
        pixel_scales=dataset.pixel_scales,
        radius=1.3,
    )
)
dataset = dataset.apply_over_sampling(
    over_sample_size_lp=1, over_sample_size_pixelization=1
)

phase("simulate")

"""
__Model + Assertion__

Two free Gaussian widths with an ORDERING ASSERTION between them. The assertion
is the whole point: it is what the latent engine's per-sample `jax.jit` used to
choke on.
"""
lens = af.Model(
    al.Galaxy,
    redshift=0.5,
    bulge=af.Model(
        al.lp.Gaussian,
        **{**GAUSSIAN_INNER, "sigma": af.GaussianPrior(mean=0.3, sigma=0.05)},
    ),
    disk=af.Model(
        al.lp.Gaussian,
        **{**GAUSSIAN_OUTER, "sigma": af.GaussianPrior(mean=0.8, sigma=0.05)},
    ),
    mass=af.Model(al.mp.Isothermal, **LENS_MASS),
    shear=af.Model(al.mp.ExternalShear, **LENS_SHEAR),
)

model = af.Collection(
    galaxies=af.Collection(
        lens=lens,
        source=af.Model(
            al.Galaxy, redshift=1.0, bulge=af.Model(al.lp.Sersic, **SOURCE_BULGE)
        ),
    )
)
model.add_assertion(
    model.galaxies.lens.bulge.sigma < model.galaxies.lens.disk.sigma,
    name="lens_gaussian_widths_ordered",
)

assert model.assertions, (
    "the model carries no assertion — this script proves nothing about the "
    "assertion-under-jit path it exists to guard."
)
assert model.total_free_parameters == 2, model.total_free_parameters

"""
__Fit__
"""
search = af.Nautilus(
    name="jax_assertion_latents",
    n_live=10,
    n_batch=10,
    n_like_max=40,
    # One neural bound instead of Nautilus's default four: the bound is only a
    # proposal, so the posterior is as valid and the search is much cheaper.
    n_networks=1,
    iterations_per_quick_update=int(1e9),
    iterations_per_full_update=int(1e9),
    # Seeded so the sampler's spread — which the sigma assertions depend on —
    # is the same on every machine and CI leg.
    seed=1,
)

search.fit(
    model=model,
    analysis=al.AnalysisImaging(dataset=dataset, use_jax=True, magzero=MAGZERO),
)

phase("jax fit: search + per-sample latent jit + zip")

"""
__On-Disk Assertions__
"""
summaries = sorted(OUTPUT_PATH.rglob("files/latent/latent_summary.json"))
assert len(summaries) == 1, (
    f"a real JAX search must write exactly one files/latent/latent_summary.json; "
    f"found {summaries}. An EMPTY list is the PyAutoLens#734 / PyAutoFit#1600 "
    "failure: the model assertion raised inside the latent engine's per-sample "
    "jax.jit for every sample, so the whole latent block was dropped."
)

summary = json.loads(summaries[0].read_text())["arguments"]

values = None
for block in ("median_pdf_sample", "max_log_likelihood_sample"):
    entry = summary.get(block)
    if entry is not None:
        values = {
            key: float(value)
            for key, value in entry["arguments"]["kwargs"]["arguments"].items()
        }
        break

assert values is not None, (
    f"{summaries[0]} carries neither a median_pdf_sample nor a "
    "max_log_likelihood_sample block."
)

assert set(values) == EXPECTED_KEYS, (
    f"JAX latent key set is {sorted(set(values))}, expected "
    f"{sorted(EXPECTED_KEYS)}. A missing key is a latent that went NaN on the "
    "JAX path and was dropped from the summary."
)

for key, value in values.items():
    assert math.isfinite(value), f"JAX latent '{key}' is not finite ({value})."
    assert value != 0.0, f"JAX latent '{key}' is exactly 0.0 — a dead latent."

"""
__No Silent Per-Sample Failures__

A partially-broken latent function does not empty the block; it drops the
offending rows and logs "the latent function raised on N of M samples". Under
`search_log: true` that warning lands in `search.log`, so the log is the only
place a HALF-failed JAX latent pass shows up.
"""
logs = sorted(OUTPUT_PATH.rglob("search.log"))
assert len(logs) == 1, (
    f"expected exactly one search.log under {OUTPUT_PATH}; found {logs}. "
    "`search_log: true` in config/output.yaml is what writes it."
)

log_text = logs[0].read_text()
assert "latent function raised on" not in log_text, (
    "search.log reports that the latent function raised on some samples:\n"
    + "\n".join(
        line for line in log_text.splitlines() if "latent function raised on" in line
    )
    + "\nUnder JAX the latent engine must skip model assertions "
    "(`latent_instance_from(..., ignore_assertions=True)`); a raise here means "
    "it is re-checking them inside the per-sample jit."
)
assert (
    "skipping latent output" not in log_text
), "search.log reports that latent output was skipped entirely."

print(
    f"PASSED: JAX search with a model assertion wrote all {len(EXPECTED_KEYS)} "
    f"latents, every value finite and non-zero, no per-sample raises in "
    "search.log"
)
for key in sorted(values):
    print(f"  {key}: {values[key]:.6g}")

phase("on-disk assertions")

print(
    "[phase] TOTAL (excluding interpreter start): "
    f"{time.perf_counter() - _START:.1f}s"
)
