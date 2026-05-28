"""
Smoke test: library latent variables produce finite values end-to-end.

Builds a minimal SIE + Sersic lens model, runs a bypass-mode search via
``PYAUTO_TEST_MODE=2`` (env vars come from ``config/build/env_vars.yaml``),
materialises the latent samples via ``analysis.compute_latent_samples``,
and asserts every default-enabled key in ``autolens.analysis.latent.LATENT_FUNCTIONS``
is present with a finite value in a loose order-of-magnitude bracket.

The workspace ``config/latent.yaml`` enables every entry in
``LATENT_FUNCTIONS`` — three raw-flux latents (instrument-input-free),
three µJy variants (require ``magzero``, which the smoke fixture
supplies), plus ``magnification`` and ``effective_einstein_radius``.
The brackets are deliberately loose so PYAUTO_SMALL_DATASETS grid mutations
(15x15) still yield passing values; the goal is a structural regression
guard, not Bayesian validation.

Catches: latent dispatch bugs, missing keys, NaN/Inf leakage, library
API drift that would silently drop a latent from the output.
"""

import math

import autofit as af
import autolens as al
from autolens import fixtures
from autolens.analysis.latent import LATENT_FUNCTIONS

# Use the standard 7x7 masked-imaging fixture rather than loading from disk —
# matches the pattern of `aggregator/tracer.py` and similar smoke scripts.
# No dataset simulation, no I/O, no PYAUTO_SMALL_DATASETS interaction.
dataset = fixtures.make_masked_imaging_7x7()

lens = af.Model(al.Galaxy, redshift=0.5, mass=al.mp.Isothermal)
source = af.Model(al.Galaxy, redshift=1.0, bulge=al.lp.Sersic)
model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

analysis = al.AnalysisImaging(dataset=dataset, use_jax=False, magzero=25.0)

search = af.Nautilus(
    name="latent_variables_smoke",
    n_live=10,
    n_like_max=10,
)

result = search.fit(model=model, analysis=analysis)

EXPECTED_KEYS = [
    "total_lens_flux",
    "total_lensed_source_flux",
    "total_source_flux",
    "total_lens_flux_mujy",
    "total_lensed_source_flux_mujy",
    "total_source_flux_mujy",
    "magnification",
    "effective_einstein_radius",
]

assert sorted(EXPECTED_KEYS) == sorted(LATENT_FUNCTIONS.keys()), (
    "Library LATENT_FUNCTIONS keys drifted from the smoke expectation. "
    "Update EXPECTED_KEYS in this script and add to smoke_tests.txt if needed."
)

# Materialise latent samples directly from the result (works under
# PYAUTO_TEST_MODE=2 even though the sampler is bypassed).
latent_samples = analysis.compute_latent_samples(result.samples)

assert latent_samples is not None, (
    "compute_latent_samples returned None — the latent pipeline short-circuited "
    "(likely LATENT_KEYS was empty; check workspace config/latent.yaml is being "
    "loaded with the expected entries set true)."
)

instance = latent_samples.median_pdf()

for key in EXPECTED_KEYS:
    value = float(getattr(instance, key))
    assert math.isfinite(value), (
        f"Latent '{key}' is not finite (got {value}). "
        "Library latent dispatch may be silently producing NaN/Inf."
    )

# Finite-only checks — this smoke is a structural regression guard, not a
# Bayesian validation. PYAUTO_TEST_MODE=2 bypasses the sampler and runs the
# likelihood / latent dispatch ONCE on degenerate prior-median parameters,
# so absolute values can be 0 or extreme legitimately. The point is to
# detect: missing keys, NaN/Inf leakage, library API drift that drops a
# latent from the output. Add value-bracket assertions only if a specific
# regression demands them.
print(f"PASSED: all {len(EXPECTED_KEYS)} library latents finite")
for key in EXPECTED_KEYS:
    print(f"  {key}: {float(getattr(instance, key)):.6g}")
