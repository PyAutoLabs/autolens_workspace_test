"""
Integration guard: latent variables that go NaN in an arbitrary per-sample
pattern must NOT crash the end-of-search latent summary.

Reproduces the production failure Sam hit on a multipoles SLaM ``mass_total``
run:

    KeyError: "Could not find any of the following keys in kwargs
               (('total_lensed_source_flux',),)"

Root cause (``PyAutoFit/autofit/non_linear/analysis/analysis.py::
compute_latent_samples``): the JAX batch path masked finite latent *columns*
**per batch** (``jnp.all(isfinite, axis=0)``). A single sample whose latent
went NaN in one batch dropped that whole latent column for that batch only,
while other batches kept it — so different samples ended up with different
``Sample.kwargs`` key sets. ``Samples.summary()`` then built its model from
batch 0's keys and raised ``KeyError`` on the first sample from a reduced
batch. ``total_lensed_source_flux`` goes NaN for real when the lensed source
image lookup fails for a degenerate source, which is easy to hit on a long
SLaM run.

How this test forces the condition deterministically: the
``PYAUTO_LATENT_NAN_INJECT=stride:N`` knob (``autonerves.test_mode``) sets NaN on
latent column 0 for every sample whose absolute index is a non-zero multiple
of ``N``. With the latent ``batch_size`` chosen so ``N >= batch_size``, batch 0
stays fully finite (seeds the model with the complete key set) and a later
batch loses column 0 — exactly the inconsistency above.

The bug lives in the JAX column-mask branch only (the NumPy branch row-masks
first, so it never produces inconsistent keys). The search itself therefore
runs on the cheap NumPy path; latents are materialised with a separate
``use_jax=True`` analysis so the JAX masking branch is the code under test.

PASS (post-fix) means ``compute_latent_samples(...).summary()`` and
``median_pdf()`` succeed and every surviving latent is finite. FAIL (pre-fix)
is the ``KeyError`` above. This is a structural regression guard, not a
Bayesian validation.
"""

import math
import os

# Must be set before the latent computation runs; read at call time inside
# autofit.compute_latent_samples. Process-scoped to this script (each smoke
# script runs in its own process), which is exactly what we want — it must NOT
# leak to other latent tests.
os.environ["PYAUTO_LATENT_NAN_INJECT"] = "stride:3"
# Skip the search's incidental post-fit latent pass (the NumPy path, which is
# not the branch under test). The explicit compute_latent_samples call below is
# NOT gated by this flag, so the JAX path we care about still runs — this just
# avoids redundantly recomputing all latents during search.fit.
os.environ["PYAUTO_SKIP_LATENTS"] = "1"

import autofit as af
import autolens as al
from autolens import fixtures
from autolens.analysis.latent import LATENT_FUNCTIONS

LATENT_BATCH_SIZE = 3  # <= the stride above, so batch 0 stays fully finite.

dataset = fixtures.make_masked_imaging_7x7()

lens = af.Model(al.Galaxy, redshift=0.5, mass=al.mp.Isothermal)
source = af.Model(al.Galaxy, redshift=1.0, bulge=al.lp.Sersic)
model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

# Search on the NumPy path (fast, reliable) — we only need a populated samples
# object. The latent masking bug is JAX-only, so latents are recomputed below
# with a use_jax=True analysis.
analysis = al.AnalysisImaging(dataset=dataset, use_jax=False, magzero=25.0)

search = af.Nautilus(
    name="latent_nan_robustness",
    n_live=15,
    n_like_max=30,
)

result = search.fit(model=model, analysis=analysis)

assert len(result.samples.sample_list) > LATENT_BATCH_SIZE, (
    f"Need >{LATENT_BATCH_SIZE} samples for a multi-batch latent run; got "
    f"{len(result.samples.sample_list)}. Increase n_live / n_like_max."
)

# Materialise latents on the JAX path — this is the branch the bug lives in.
analysis_jax = al.AnalysisImaging(dataset=dataset, use_jax=True, magzero=25.0)

latent_samples = analysis_jax.compute_latent_samples(
    result.samples, batch_size=LATENT_BATCH_SIZE
)

assert latent_samples is not None, (
    "compute_latent_samples returned None — the latent pipeline short-circuited. "
    "Expected a populated latent Samples object."
)

# These two calls are what crash pre-fix (KeyError in parameter_lists_for_paths).
summary = latent_samples.summary()
instance = latent_samples.median_pdf()

# Every latent that survived global masking must be finite. We do not require
# the injected latent (total_lens_flux, column 0) to survive — global masking
# may legitimately drop samples carrying its NaN — but whatever remains must be
# clean, and in particular the summary must have been built without a KeyError.
surviving = [k for k in LATENT_FUNCTIONS if hasattr(instance, k)]
assert surviving, "No latents survived — expected at least the finite raw-flux latents."

for key in surviving:
    value = float(getattr(instance, key))
    assert math.isfinite(
        value
    ), f"Surviving latent '{key}' is not finite (got {value})."

print(
    f"PASSED: latent summary survived arbitrary NaN injection "
    f"({len(surviving)} latents finite, batch_size={LATENT_BATCH_SIZE})."
)
for key in surviving:
    print(f"  {key}: {float(getattr(instance, key)):.6g}")
