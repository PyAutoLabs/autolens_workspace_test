"""
JAX Reproducer: Free Subhalo Redshift Triggers TracerBoolConversionError
========================================================================

This script is the integration-test reproducer for issue
https://github.com/PyAutoLabs/PyAutoLens/issues/498, reported by @qiuhan96.

When a galaxy named ``subhalo`` is added to the model and its ``redshift`` is
made a free parameter (an ``af.UniformPrior``), the JAX path of
``AnalysisImaging`` raises ``jax.errors.TracerBoolConversionError``. The error
originates in ``autolens/lens/tracer_util.py`` (``plane_redshifts_from`` and
``grid_2d_at_redshift_from``), which perform Python-level ``<``, ``<=``, ``==``
and ``float()`` on values that under JIT are traced scalars.

This script runs two scenarios so the failure mode is unambiguous:

- **Scenario A** — subhalo redshift is a fixed Python float (z=0.55). The JAX
  path should succeed (vmap + jit-wrapped ``analysis.fit_from``).
- **Scenario B** — subhalo redshift is ``af.UniformPrior(0.2, 0.9)``. The JAX
  path is expected to raise ``TracerBoolConversionError``. A trimmed traceback
  is printed and the script reports ``REPRODUCED: issue #498``.

Both scenarios use the same ``dataset/imaging/jax_test`` dataset that
``lp.py`` uses, the same lens (PowerLaw + ExternalShear + linear Sersic
bulge) and the same source (linear Sersic). Only the subhalo galaxy and the
type of its ``redshift`` differ.

Once issue #498 is fixed, Scenario B will stop raising. The script will then
print ``UNEXPECTED: bug appears fixed`` and exit nonzero, which surfaces in
``run_all_scripts.sh`` so we know to flip the assertion polarity.
"""

# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import sys
import traceback

import numpy as np
import jax.numpy as jnp
import jax
from os import path

import autofit as af
import autolens as al

"""
__Dataset__

Reuse the ``jax_test`` dataset that ``lp.py`` uses (lens at z=0.5,
source at z=1.0). If it does not exist on disk, run the same
auto-simulation fallback as ``lp.py``.
"""
dataset_path = path.join("dataset", "imaging", "jax_test")

if al.util.dataset.should_simulate(dataset_path):
    import subprocess

    subprocess.run(
        [sys.executable, "scripts/jax_likelihood_functions/imaging/simulator.py"],
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
"""
mask_radius = 3.0

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=mask_radius,
)

dataset = dataset.apply_mask(mask=mask)
dataset = dataset.apply_over_sampling(over_sample_size_lp=1)

positions = al.Grid2DIrregular(
    al.from_json(file_path=path.join(dataset_path, "positions.json"))
)


"""
__Model Builder__

A helper that builds the full ``af.Collection`` for a given subhalo redshift.
The lens and source pieces match ``lp.py``. The subhalo composition matches
the bug report from @qiuhan96 (``mp.IsothermalSph`` with the same prior
ranges he supplied), with the galaxy attribute deliberately named
``subhalo`` so that ``AnalysisLens.tracer_via_instance_from`` enters the
buggy branch at ``analysis/lens.py:105`` (``hasattr(instance.galaxies,
"subhalo")``).
"""


def build_model(redshift_subhalo):
    lens = af.Model(
        al.Galaxy,
        redshift=0.5,
        bulge=af.Model(al.lp_linear.Sersic),
        mass=af.Model(al.mp.PowerLaw),
        shear=af.Model(al.mp.ExternalShear),
    )

    subhalo_mass = af.Model(al.mp.IsothermalSph)
    subhalo_mass.centre_0 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
    subhalo_mass.centre_1 = af.UniformPrior(lower_limit=1.2, upper_limit=1.8)
    subhalo_mass.einstein_radius = af.UniformPrior(lower_limit=0.01, upper_limit=0.4)

    subhalo = af.Model(al.Galaxy, redshift=redshift_subhalo, mass=subhalo_mass)

    source = af.Model(
        al.Galaxy,
        redshift=1.0,
        bulge=af.Model(al.lp_linear.Sersic),
    )

    return af.Collection(
        galaxies=af.Collection(lens=lens, subhalo=subhalo, source=source)
    )


"""
__Scenario Runner__

Builds an ``AnalysisImaging`` (with positions likelihood, mirroring ``lp.py``),
runs ``fitness._vmap`` over a small batch of prior-median parameter vectors,
then jit-wraps ``analysis.fit_from`` and runs it on a single instance.

Returns ``(ok, exc)`` so the caller can decide whether the result is what was
expected for that scenario.
"""

from autofit.non_linear.fitness import Fitness
from autofit.jax.pytrees import enable_pytrees, register_model

# enable_pytrees once globally so both scenarios benefit from it for the
# Path-A jit wrap.  ``register_model`` is called per-scenario below.
enable_pytrees()


def run_scenario(label, redshift_subhalo, batch_size=4):
    print()
    print("=" * 72)
    print(f"Scenario {label}: redshift_subhalo = {redshift_subhalo!r}")
    print("=" * 72)

    model = build_model(redshift_subhalo)
    register_model(model)

    analysis = al.AnalysisImaging(
        dataset=dataset,
        positions_likelihood_list=[
            al.PositionsLH(threshold=0.4, positions=positions)
        ],
    )

    # --- Path 1: fitness._vmap over a small batch of prior-median vectors ---
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

    try:
        result = fitness._vmap(parameters)
        print(f"  [vmap]   result shape={np.shape(np.array(result))}, "
              f"first={float(np.array(result)[0]):.6e}")
    except Exception as exc:  # noqa: BLE001 — diagnostic, want to keep going
        traceback.print_exc()
        return False, exc

    # --- Path 2: jit-wrapped analysis.fit_from on a single instance ---
    instance = model.instance_from_prior_medians()

    analysis_np = al.AnalysisImaging(
        dataset=dataset,
        positions_likelihood_list=[
            al.PositionsLH(threshold=0.4, positions=positions)
        ],
        use_jax=False,
    )
    fit_np = analysis_np.fit_from(instance=instance)
    print(f"  [numpy]  fit.log_likelihood = {float(fit_np.log_likelihood):.6e}")

    analysis_jit = al.AnalysisImaging(
        dataset=dataset,
        positions_likelihood_list=[
            al.PositionsLH(threshold=0.4, positions=positions)
        ],
        use_jax=True,
    )
    try:
        fit = jax.jit(analysis_jit.fit_from)(instance)
        print(f"  [jit]    fit.log_likelihood = {float(fit.log_likelihood):.6e}")
        np.testing.assert_allclose(
            float(fit.log_likelihood), float(fit_np.log_likelihood), rtol=1e-4
        )
        print(f"  [jit]    matches numpy path within rtol=1e-4")
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        return False, exc

    return True, None


"""
__Scenario A — Fixed Subhalo Redshift__

The subhalo redshift is a Python float between the lens (z=0.5) and source
(z=1.0). This exercises the same multi-plane code path as Scenario B but
without any traced-scalar comparisons, so it should succeed.
"""
ok_a, exc_a = run_scenario("A (fixed redshift z=0.55)", redshift_subhalo=0.55)


"""
__Scenario B — Free Subhalo Redshift__

The subhalo redshift is an ``af.UniformPrior(0.2, 0.9)``. Under JAX this
becomes a traced scalar, and ``tracer_util`` then performs Python boolean
comparisons that JAX cannot lift to traced ops.
"""
ok_b, exc_b = run_scenario(
    "B (free redshift UniformPrior(0.2, 0.9))",
    redshift_subhalo=af.UniformPrior(lower_limit=0.2, upper_limit=0.9),
)


"""
__Verdict__
"""
print()
print("=" * 72)
print("Summary")
print("=" * 72)
print(f"  Scenario A (fixed)  : {'PASS' if ok_a else 'FAIL'}")
print(f"  Scenario B (free)   : {'PASS' if ok_b else 'FAIL (raised)'}")

if not ok_a:
    print()
    print(f"FAIL: Scenario A unexpectedly raised: {type(exc_a).__name__}: {exc_a}")
    sys.exit(1)

if ok_b:
    print()
    print("UNEXPECTED: Scenario B did not raise — bug from issue #498 may be fixed.")
    print("Update the issue and flip this script's expected outcome.")
    sys.exit(1)

# Scenario B raised, which is what we expect today.
exc_name = type(exc_b).__name__
print()
print(f"REPRODUCED: issue #498")
print(f"  Scenario B raised {exc_name}: {exc_b}")
print()
