"""
Integration smoke: latents are computed by a REAL search, written to disk, and
read back through the aggregator and ``af.AggregateCSV``.
=========================================================================

The sibling ``latent_variables_smoke.py`` runs under the smoke profile's
``PYAUTO_TEST_MODE=2``, where ``autonerves.test_mode.skip_latents()`` suppresses
all latent *writing* (``PyAutoFit/autofit/non_linear/search/updater.py``). It
hand-calls ``analysis.compute_latent_samples`` and therefore never touches
``files/latent/latent_summary.json``, the aggregator, or ``af.AggregateCSV`` —
the exact path the Euclid catalogue builds on. Four latent defects shipped or
sat unnoticed between June and September 2026 because nothing exercised it:

 - the retired ``latent.`` argument prefix silently produced blank catalogue
   columns (pipeline#65);
 - ``AggregateCSV`` wrote the 1-sigma values into every latent 3-sigma column
   and seeded max-likelihood from the median (PyAutoFit#1598);
 - ``magnification`` was ``inf``/``0.0`` for every pixelized source
   (PyAutoLens#727/#728);
 - a sibling ``<hash>/`` directory beside a ``<hash>.zip`` made the aggregator
   drop that search entirely (PyAutoFit#1601, fixed by #1602).

This script closes that gap. It runs two REAL searches on a small in-script
simulation and asserts on what actually lands on disk:

Stage 1 (``stage_1_lp``) — ``af.Nautilus`` on a light-profile source. Produces a
real PDF, so the latent summary carries a median sample, 1-sigma and 3-sigma
values.

Stage 2 (``stage_2_pix``) — ``af.Drawer``, the cheapest real search, on a
``RectangularUniform`` + ``Constant`` pixelized source. This is the only way to
exercise ``_pixelized_source_flux``, and hence ``magnification``, on the write
path. ``Drawer`` returns a plain ``Samples`` with no PDF, so only the
max-likelihood columns can be non-blank on its row — asserted explicitly.

Models here are composed as ``af.Model(cls, **fixed_values)``, never
``af.Model.from_instance(...)``: ``from_instance`` copies DERIVED attributes onto
the model (``Isothermal.slope``, ``ExternalShear.centre``/``ell_comps``) that the
class ``__init__`` does not accept, so the ``model.json`` such a search writes
cannot be deserialised and the aggregator raises ``TypeError`` before it can read
a single result. Reported separately; it is a library defect, not something this
script should paper over.

Both searches run with ``output.remove_files: true``, so each result survives
only inside its ``<identifier>.zip`` and the aggregator must be driven with
``unzip_temporary=True``.

__Env__ (Developer Only)

``real_search`` is load-bearing: ``skip_latents()`` is true for ANY
``PYAUTO_TEST_MODE`` level, so under the smoke profile's default of ``2`` no
latent block is written at all and every assertion below would be vacuous. The
script asserts ``not skip_latents()`` up front so a profile regression fails
loudly rather than passing on an empty tree.

``full_datasets`` is load-bearing too: under ``PYAUTO_SMALL_DATASETS=1`` grids
and masks are capped to 15x15, which is too coarse for
``LensCalc.einstein_radius_from`` to resolve the tangential critical curve —
``effective_einstein_radius`` would come back NaN and be DROPPED from the
summary, so the exact-key-set assertion would fail.

ENV: real_search full_datasets
"""

import json
import math
import os
import shutil
import time
import zipfile
from pathlib import Path

from autonerves.test_mode import skip_latents

WORKSPACE = Path(__file__).resolve().parents[3]

"""
__Guards__

Both tokens above must have taken effect; otherwise this script proves nothing.
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

"""
__Config Overlay__

A copy of the workspace `config/` with three keys changed, pushed via
`conf.instance.push` so the repository's own config is never mutated:

 - `output.latent_draw_via_pdf: false` — latents are computed from every stored
   sample rather than a handful of PDF draws. This is what makes the 3-sigma
   bracket assertion meaningful (8 PDF draws give degenerate tails) and it is
   also the branch that writes `files/latent/samples.csv` alongside
   `latent_summary.json`.
 - `output.latent_csv: true` — the `latent.csv` file (already the workspace
   default; pinned here so a config change cannot silently drop it).
 - `general.output.remove_files: true` — the search zips its output and deletes
   the unzipped tree, so the aggregator has to go through the zip path.
"""
OUTPUT_PATH = WORKSPACE / "output" / "latent_integration_smoke"
CONFIG_PATH = WORKSPACE / "output" / "latent_integration_smoke_config"

# A warm output tree fakes NoneType failures on re-run; wipe both first.
for stale in (OUTPUT_PATH, CONFIG_PATH):
    if stale.exists():
        shutil.rmtree(stale)

shutil.copytree(WORKSPACE / "config", CONFIG_PATH)

output_yaml = CONFIG_PATH / "output.yaml"
output_yaml.write_text(
    output_yaml.read_text()
    .replace("latent_draw_via_pdf : true", "latent_draw_via_pdf : false")
    .replace("latent_csv: false", "latent_csv: true")
)
assert "latent_draw_via_pdf : false" in output_yaml.read_text()

general_yaml = CONFIG_PATH / "general.yaml"
general_text = general_yaml.read_text()
assert "remove_files: false" in general_text, (
    "config/general.yaml no longer carries `remove_files: false`; the overlay "
    "edit below is a no-op and the zip path would go untested."
)
general_yaml.write_text(
    general_text.replace("remove_files: false", "remove_files: true")
)

from autolens import conf  # noqa: E402  (must precede the autofit/autolens import)

conf.instance.push(new_path=CONFIG_PATH, output_path=OUTPUT_PATH)

import autofit as af  # noqa: E402
import autolens as al  # noqa: E402

# `af.Aggregator` is the DATABASE aggregator; the output-directory one is
# imported by its module path, as `autolens_workspace/guides/results` does.
from autofit.aggregator.aggregator import Aggregator  # noqa: E402
from autolens.analysis.latent import LATENT_FUNCTIONS  # noqa: E402

"""
__Enabled Latent Keys__

The key set the workspace `config/latent.yaml` switches on. A latent that comes
back NaN is DROPPED from `latent_summary.json` (a missing key, not a null), so
asserting the exact set is what catches a silently-dead latent.
"""
EXPECTED_KEYS = set(LATENT_FUNCTIONS)
enabled = conf.instance["latent"]
assert EXPECTED_KEYS == {key for key in LATENT_FUNCTIONS if enabled[key]}, (
    "config/latent.yaml no longer enables every entry in LATENT_FUNCTIONS; this "
    "script asserts the full registry is written."
)

MAGZERO = 25.0

"""
__Simulate__

A 40x40 image at 0.1"/pixel: an SIE (theta_E = 1.0") + shear lens with a Sersic
bulge, and a Sersic source. 4" across is wide enough for
`LensCalc.einstein_radius_from` to resolve the tangential critical curve at this
theta_E.
"""
t0 = time.perf_counter()

grid = al.Grid2D.uniform(shape_native=(40, 40), pixel_scales=0.1)

psf = al.Convolver.from_gaussian(
    shape_native=(11, 11),
    sigma=0.1,
    pixel_scales=grid.pixel_scales,
    convolve_over_sample_size=1,
)

# One parameter set builds both the truth instances and the models below, so a
# prior can never drift away from the value it is meant to be anchored on.
#
# Models are composed as `af.Model(cls, **fixed)` rather than
# `af.Model.from_instance(galaxy)`: from_instance copies DERIVED attributes onto
# the model (`Isothermal.slope`, `ExternalShear.centre`/`ell_comps`), which the
# class `__init__` does not accept, so the `model.json` such a search writes
# cannot be read back and the aggregator raises `TypeError` on it. See the note
# in the module docstring.
LENS_BULGE = dict(
    centre=(0.0, 0.0),
    ell_comps=(0.0, 0.0),
    intensity=1.0,
    effective_radius=0.6,
    sersic_index=3.0,
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

TRUTH_LENS = al.Galaxy(
    redshift=0.5,
    bulge=al.lp.Sersic(**LENS_BULGE),
    mass=al.mp.Isothermal(**LENS_MASS),
    shear=al.mp.ExternalShear(**LENS_SHEAR),
)

TRUTH_SOURCE = al.Galaxy(redshift=1.0, bulge=al.lp.Sersic(**SOURCE_BULGE))

simulator = al.SimulatorImaging(
    exposure_time=300.0,
    psf=psf,
    background_sky_level=0.1,
    add_poisson_noise_to_data=True,
    noise_seed=1,
)

dataset = simulator.via_tracer_from(
    tracer=al.Tracer(galaxies=[TRUTH_LENS, TRUTH_SOURCE]), grid=grid
)

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=1.6,
)
dataset = dataset.apply_mask(mask=mask)

print(f"[timing] simulate: {time.perf_counter() - t0:.1f}s")

"""
__Stage 1: Light-Profile Source (Nautilus)__

Truth-anchored Gaussian priors on three parameters, everything else fixed. A
real sampler is required: only a search with a PDF gives the latent summary a
median sample and 1/3-sigma values.
"""


def lens_model(einstein_radius, intensity):
    """
    The lens fixed to truth apart from the two parameters the lens-side latents
    are sensitive to: the Einstein radius (`effective_einstein_radius`) and the
    bulge intensity (`total_lens_flux`). Either may be passed a prior (free) or
    a float (fixed).
    """
    return af.Model(
        al.Galaxy,
        redshift=0.5,
        bulge=af.Model(al.lp.Sersic, **{**LENS_BULGE, "intensity": intensity}),
        mass=af.Model(
            al.mp.Isothermal, **{**LENS_MASS, "einstein_radius": einstein_radius}
        ),
        shear=af.Model(al.mp.ExternalShear, **LENS_SHEAR),
    )


source_lp = af.Model(
    al.Galaxy,
    redshift=1.0,
    bulge=af.Model(
        al.lp.Sersic,
        **{**SOURCE_BULGE, "intensity": af.GaussianPrior(mean=2.0, sigma=0.2)},
    ),
)

model_1 = af.Collection(
    galaxies=af.Collection(
        lens=lens_model(
            einstein_radius=af.GaussianPrior(mean=1.0, sigma=0.05),
            intensity=af.GaussianPrior(mean=1.0, sigma=0.1),
        ),
        source=source_lp,
    )
)
assert model_1.total_free_parameters == 3, model_1.total_free_parameters

t0 = time.perf_counter()

search_1 = af.Nautilus(
    name="stage_1_lp",
    n_live=50,
    n_batch=50,
    n_like_max=800,
    iterations_per_quick_update=int(1e9),
    iterations_per_full_update=int(1e9),
    # Seeded so the sampler's spread — which the sigma assertions depend on —
    # is the same on every machine and CI leg.
    seed=1,
)

result_1 = search_1.fit(
    model=model_1,
    analysis=al.AnalysisImaging(dataset=dataset, use_jax=False, magzero=MAGZERO),
)

stage_1_secs = time.perf_counter() - t0
print(f"[timing] stage_1_lp (Nautilus, real search): {stage_1_secs:.1f}s")

"""
__Stage 2: Pixelized Source (Drawer)__

`Galaxy.image_2d_from` returns zeros for a source whose only light model is a
`Pixelization`, so `magnification` is finite only if `_pixelized_source_flux`
integrates the reconstruction correctly (PyAutoLens#726/#727/#728). `af.Drawer`
is the cheapest real search that still writes a latent block.
"""
source_pix = af.Model(
    al.Galaxy,
    redshift=1.0,
    pixelization=af.Model(
        al.Pixelization,
        mesh=al.mesh.RectangularUniform(shape=(10, 10)),
        regularization=af.Model(
            al.reg.Constant,
            coefficient=af.GaussianPrior(mean=1.0, sigma=0.2),
        ),
    ),
)

# The lens is fixed to truth apart from the Einstein radius, which is re-fit
# inside a narrow prior around the stage-1 median. Keeping one MODEL parameter
# free in both stages is what lets the catalogue below carry a model column that
# resolves on both rows under `strict=True`.
stage_1_einstein_radius = float(
    result_1.samples.median_pdf().galaxies.lens.mass.einstein_radius
)
model_2 = af.Collection(
    galaxies=af.Collection(
        lens=lens_model(
            einstein_radius=af.GaussianPrior(mean=stage_1_einstein_radius, sigma=0.01),
            intensity=LENS_BULGE["intensity"],
        ),
        source=source_pix,
    )
)
assert model_2.total_free_parameters == 2, model_2.total_free_parameters

t0 = time.perf_counter()

search_2 = af.Drawer(name="stage_2_pix", total_draws=6)

result_2 = search_2.fit(
    model=model_2,
    analysis=al.AnalysisImaging(dataset=dataset, use_jax=False, magzero=MAGZERO),
)

stage_2_secs = time.perf_counter() - t0
print(f"[timing] stage_2_pix (Drawer, pixelized source): {stage_2_secs:.1f}s")

"""
__On-Disk Assertions__

Everything below reads what the searches WROTE, not what they returned. Under
`remove_files: true` each result survives only inside `<identifier>.zip`.
"""
MODEL_PATH = "galaxies.lens.mass.einstein_radius"


def stage_zip(stage_name):
    """
    The single `<identifier>.zip` a completed stage leaves behind, and its
    identifier (which is also the row id in the catalogue below).
    """
    zips = sorted((OUTPUT_PATH / stage_name).glob("*.zip"))
    assert len(zips) == 1, (
        f"{stage_name} must leave exactly one search-output zip under "
        f"`remove_files: true`; found {zips}"
    )
    unzipped = [d for d in (OUTPUT_PATH / stage_name).iterdir() if d.is_dir()]
    assert not unzipped, (
        f"{stage_name} left an unzipped tree ({unzipped}) — `remove_files: true` "
        "did not take effect, so the zip read path below is not being tested."
    )
    return zips[0], zips[0].stem


def latent_summary_from(stage_zip_path):
    """
    The `arguments` block of `files/latent/latent_summary.json`, read out of the
    zip. Also checks the latent `samples.csv` landed: it is written only on the
    `latent_draw_via_pdf: false` branch, which is the branch the overlay selects.
    """
    with zipfile.ZipFile(stage_zip_path) as archive:
        names = archive.namelist()
        assert "files/latent/latent_summary.json" in names, (
            f"{stage_zip_path} carries no files/latent/latent_summary.json — the "
            "search wrote no latent block at all. This is the failure "
            "PyAutoLens#734 / PyAutoFit#1600 shipped."
        )
        assert "files/latent/samples.csv" in names, (
            f"{stage_zip_path} carries no files/latent/samples.csv; the "
            "`latent_draw_via_pdf: false` write branch did not run."
        )
        return json.loads(archive.read("files/latent/latent_summary.json"))["arguments"]


def sample_values(summary, block):
    """The latent name -> value mapping of one sample block, or None."""
    entry = summary.get(block)
    if entry is None:
        return None
    return {
        key: float(value)
        for key, value in entry["arguments"]["kwargs"]["arguments"].items()
    }


def sigma_values(summary, block):
    """The latent name -> (lower, upper) mapping of one sigma block, or None."""
    entry = summary.get(block)
    if entry is None:
        return None
    return {
        key: tuple(float(v) for v in value["values"])
        for key, value in entry["arguments"].items()
    }


def assert_full_and_finite(values, where):
    """
    Exactly the enabled key set, every value finite and non-zero.

    A latent that evaluates to NaN is DROPPED from the summary — it is a MISSING
    key, not a null — so the exact-set check is what catches a silently dead
    latent (`magnification` was `inf`/`0.0` for pixelized sources for months,
    PyAutoLens#727/#728).
    """
    assert set(values) == EXPECTED_KEYS, (
        f"{where}: latent key set is {sorted(set(values))}, expected "
        f"{sorted(EXPECTED_KEYS)}. Missing keys are latents that went NaN and "
        "were dropped; extra keys mean config/latent.yaml drifted."
    )
    for key, value in values.items():
        assert math.isfinite(value), f"{where}: latent '{key}' is not finite ({value})."
        assert value != 0.0, (
            f"{where}: latent '{key}' is exactly 0.0 — a dead latent, not a "
            "measurement."
        )


stage_1_zip, stage_1_id = stage_zip("stage_1_lp")
stage_2_zip, _ = stage_zip("stage_2_pix")

summary_1 = latent_summary_from(stage_1_zip)
summary_2 = latent_summary_from(stage_2_zip)

median_1 = sample_values(summary_1, "median_pdf_sample")
max_lh_1 = sample_values(summary_1, "max_log_likelihood_sample")
assert (
    median_1 is not None
), "stage_1_lp is a real sampler and must write a median_pdf_sample block."
assert_full_and_finite(median_1, "stage_1_lp median_pdf_sample")
assert_full_and_finite(max_lh_1, "stage_1_lp max_log_likelihood_sample")

sigma_1_1 = sigma_values(summary_1, "values_at_sigma_1")
sigma_3_1 = sigma_values(summary_1, "values_at_sigma_3")
assert set(sigma_1_1) == EXPECTED_KEYS and set(sigma_3_1) == EXPECTED_KEYS

for key in sorted(EXPECTED_KEYS):
    (lower_1, upper_1), (lower_3, upper_3) = sigma_1_1[key], sigma_3_1[key]
    assert lower_1 < upper_1, f"stage_1_lp '{key}': 1-sigma bounds not ordered."
    assert lower_3 < lower_1, (
        f"stage_1_lp '{key}': lower 3-sigma {lower_3} is not below lower 1-sigma "
        f"{lower_1}. The 3-sigma block is a copy of the 1-sigma block "
        "(PyAutoFit#1598)."
    )
    assert upper_3 > upper_1, (
        f"stage_1_lp '{key}': upper 3-sigma {upper_3} is not above upper 1-sigma "
        f"{upper_1}. The 3-sigma block is a copy of the 1-sigma block "
        "(PyAutoFit#1598)."
    )

"""
Stage 2 is an MLE search: `af.Drawer` has no PDF, so the summary carries a
max-likelihood sample and NOTHING else. Asserted explicitly, because the
catalogue assertions further down rely on exactly that shape.
"""
max_lh_2 = sample_values(summary_2, "max_log_likelihood_sample")
assert_full_and_finite(max_lh_2, "stage_2_pix max_log_likelihood_sample")

assert sample_values(summary_2, "median_pdf_sample") is None, (
    "af.Drawer is an MLE search with no PDF; a median_pdf_sample block means "
    "the search API changed and the blank-cell assertions below are wrong."
)
assert sigma_values(summary_2, "values_at_sigma_1") is None
assert sigma_values(summary_2, "values_at_sigma_3") is None

assert max_lh_2["magnification"] > 0.0, (
    f"stage_2_pix magnification is {max_lh_2['magnification']}. A source whose "
    "only light model is a Pixelization has zero light-profile flux, so a "
    "non-positive magnification means `_pixelized_source_flux` failed to "
    "integrate the reconstruction (PyAutoLens#726/#727/#728)."
)
assert max_lh_2["total_source_flux"] > 0.0, (
    f"stage_2_pix total_source_flux is {max_lh_2['total_source_flux']} — the "
    "pixelized reconstruction integrated to a non-positive flux, which makes "
    "`magnification` meaningless."
)

print(
    f"PASSED: on-disk latent blocks — {len(EXPECTED_KEYS)} keys per stage, "
    f"3-sigma strictly brackets 1-sigma on stage_1_lp, "
    f"stage_2_pix magnification = {max_lh_2['magnification']:.4g}"
)

"""
__Catalogue Path: Aggregator + AggregateCSV__

The path the Euclid catalogue builds on. `unzip_temporary=True` is mandatory
here: without it the aggregator extracts each zip in place and litters the
output tree for the next run (which is also the sibling-directory condition
regression-tested at the end of this script).
"""
t0 = time.perf_counter()


def aggregator():
    return Aggregator.from_directory(
        OUTPUT_PATH, completed_only=True, unzip_temporary=True
    )


agg = aggregator()
assert len(agg) == 2, (
    f"the aggregator found {len(agg)} completed searches under {OUTPUT_PATH}, "
    "expected 2 (stage_1_lp and stage_2_pix)."
)

LATENT_COLUMNS = [
    "effective_einstein_radius",
    "magnification",
    "total_lens_flux",
    "total_source_flux",
]
VALUE_TYPES = [
    af.ValueType.Median,
    af.ValueType.MaxLogLikelihood,
    af.ValueType.ValuesAt1Sigma,
    af.ValueType.ValuesAt3Sigma,
]

summary_csv = OUTPUT_PATH / "latent_catalogue.csv"

# Rows are keyed by the search NAME, carried through as a label column derived
# from the aggregator itself. `SearchOutput.id` is recomputed from the loaded
# search + model and does NOT equal the on-disk identifier directory, and the
# aggregator's iteration order is not a stable contract, so neither is safe to
# index rows by.
stage_labels = [output.value("search").name for output in agg]
assert sorted(stage_labels) == ["stage_1_lp", "stage_2_pix"], stage_labels

aggregate_csv = af.AggregateCSV(aggregator=agg, strict=True)
aggregate_csv.add_label_column("stage", stage_labels)
for column in LATENT_COLUMNS:
    aggregate_csv.add_variable(column, value_types=VALUE_TYPES)
aggregate_csv.add_variable(MODEL_PATH, value_types=VALUE_TYPES)
aggregate_csv.save(summary_csv)

"""
Reload from disk — a catalogue that is only ever inspected in memory does not
prove the columns survived serialisation.
"""
import csv  # noqa: E402

with open(summary_csv, newline="") as f:
    rows = {row["stage"]: row for row in csv.DictReader(f)}

assert set(rows) == {
    "stage_1_lp",
    "stage_2_pix",
}, f"catalogue rows are keyed {sorted(rows)}, expected one row per stage."

row_1, row_2 = rows["stage_1_lp"], rows["stage_2_pix"]

"""
Stage 1 came from a real sampler, so EVERY cell on its row is populated. A blank
cell here is the `latent.`-prefix class of defect: the column was requested, the
name did not resolve, and the catalogue shipped silent blanks (pipeline#65).
"""
blank_1 = sorted(name for name, value in row_1.items() if value in ("", None))
assert not blank_1, (
    f"stage_1_lp catalogue row has blank cells: {blank_1}. Every requested "
    "column must resolve for a search with a full PDF."
)

for column in LATENT_COLUMNS:
    median = float(row_1[column])
    max_lh = float(row_1[f"{column}_max_lh"])
    lower_1, upper_1 = float(row_1[f"{column}_lower_1_sigma"]), float(
        row_1[f"{column}_upper_1_sigma"]
    )
    lower_3, upper_3 = float(row_1[f"{column}_lower_3_sigma"]), float(
        row_1[f"{column}_upper_3_sigma"]
    )

    assert median == median_1[column], (
        f"catalogue Median for '{column}' ({median}) does not match the "
        f"median_pdf_sample in latent_summary.json ({median_1[column]})."
    )
    assert max_lh == max_lh_1[column], (
        f"catalogue MaxLogLikelihood for '{column}' ({max_lh}) does not match the "
        f"max_log_likelihood_sample in latent_summary.json ({max_lh_1[column]}). "
        "The max-likelihood column was seeded from the median (PyAutoFit#1598)."
    )
    assert (lower_1, upper_1) == sigma_1_1[column], (
        f"catalogue 1-sigma cells for '{column}' ({lower_1}, {upper_1}) do not "
        f"match values_at_sigma_1 in latent_summary.json ({sigma_1_1[column]})."
    )
    assert (lower_3, upper_3) == sigma_3_1[column], (
        f"catalogue 3-sigma cells for '{column}' ({lower_3}, {upper_3}) do not "
        f"match values_at_sigma_3 in latent_summary.json ({sigma_3_1[column]}). "
        "The 3-sigma columns were filled from the wrong block (PyAutoFit#1598)."
    )
    assert lower_3 < lower_1 and upper_3 > upper_1, (
        f"catalogue 3-sigma bracket for '{column}' does not enclose the 1-sigma "
        f"bracket: 3-sigma ({lower_3}, {upper_3}) vs 1-sigma ({lower_1}, "
        f"{upper_1}). This is exactly PyAutoFit#1598, where the 1-sigma values "
        "were written into the 3-sigma columns."
    )

assert any(
    float(row_1[column]) != float(row_1[f"{column}_max_lh"])
    for column in LATENT_COLUMNS
), (
    "every latent's MaxLogLikelihood cell equals its Median cell — the "
    "max-likelihood column is a copy of the median column (PyAutoFit#1598)."
)

"""
Stage 2 is `af.Drawer`: max-likelihood cells populated, median and sigma cells
blank. Asserting the blanks (rather than tolerating them) is what would catch a
regression that starts filling them from the wrong sample.
"""
for column in LATENT_COLUMNS + [MODEL_PATH.replace(".", "_")]:
    assert row_2[f"{column}_max_lh"] not in ("", None), (
        f"stage_2_pix catalogue cell '{column}_max_lh' is blank; an MLE search "
        "still has a max-likelihood sample."
    )
    for suffix in (
        "",
        "_lower_1_sigma",
        "_upper_1_sigma",
        "_lower_3_sigma",
        "_upper_3_sigma",
    ):
        assert row_2[f"{column}{suffix}"] in ("", None), (
            f"stage_2_pix catalogue cell '{column}{suffix}' is populated, but "
            "af.Drawer has no PDF and therefore no median or sigma values."
        )

for column in LATENT_COLUMNS:
    assert float(row_2[f"{column}_max_lh"]) == max_lh_2[column]

print(
    f"PASSED: catalogue — {len(rows)} rows, "
    f"{len(row_1)} populated cells on the stage_1_lp row, "
    f"stage_2_pix max-likelihood cells populated and PDF cells blank "
    f"({time.perf_counter() - t0:.1f}s)"
)

"""
__Regression: the retired `latent.` prefix__

Latent keys are BARE names. The Euclid catalogue asked for `latent.<name>` and
`AggregateCSV` (non-strict by default) answered with a warning and blank columns
for months (pipeline#65). `strict=True` is what turns that into a KeyError.
"""
prefixed = af.AggregateCSV(aggregator=agg, strict=True)
prefixed.add_variable(f"latent.{LATENT_COLUMNS[0]}")

try:
    prefixed.save(OUTPUT_PATH / "prefixed_should_not_exist.csv")
except KeyError:
    pass
else:
    raise AssertionError(
        f"AggregateCSV(strict=True) accepted 'latent.{LATENT_COLUMNS[0]}'. The "
        "retired prefix must raise, not silently produce a blank column."
    )

lenient = af.AggregateCSV(aggregator=agg, strict=False)
lenient.add_variable(f"latent.{LATENT_COLUMNS[0]}")
lenient_csv = OUTPUT_PATH / "prefixed_lenient.csv"
lenient.save(lenient_csv)

with open(lenient_csv, newline="") as f:
    lenient_rows = list(csv.DictReader(f))

assert all(
    row[f"latent_{LATENT_COLUMNS[0]}"] in ("", None) for row in lenient_rows
), "the non-strict path must warn and leave the unresolved column blank."

print("PASSED: the retired `latent.` prefix raises under strict=True")

"""
__Regression: a sibling directory must not shadow a zip__

A `<hash>/` directory beside `<hash>.zip` without a `.completed` marker made the
aggregator drop that search entirely, so a catalogue silently lost rows
(PyAutoFit#1601). The condition arises in practice whenever an aggregator has
been run without `unzip_temporary=True` and then interrupted.
"""
del agg  # release the temporary extraction before re-scanning the tree

sibling = stage_1_zip.parent / stage_1_id / "files"
sibling.mkdir(parents=True)
(sibling / "dummy.json").write_text("{}")

agg_sibling = aggregator()
assert len(agg_sibling) == 2, (
    f"a sibling '{stage_1_id}/' directory beside the stage-1 zip reduced the "
    f"aggregator to {len(agg_sibling)} search(es). The zip must still be used "
    "(PyAutoFit#1601)."
)

print("PASSED: a sibling directory does not shadow the completed zip")
print(f"[timing] stage_1_lp {stage_1_secs:.1f}s | stage_2_pix {stage_2_secs:.1f}s")
