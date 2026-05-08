"""
JAX Regression: Free Subhalo Redshift Under jax.jit
===================================================

This script is the JAX regression check for
https://github.com/PyAutoLabs/PyAutoLens/issues/498 (originally reported by
@qiuhan96 and fixed in PyAutoLens PR #499). When the bug was present, setting
``subhalo.redshift = af.UniformPrior(...)`` raised
``jax.errors.TracerBoolConversionError`` because Python ``sorted`` /
``float()`` / ``<=`` / ``==`` were called on what became a traced scalar
under ``jax.jit``. The fix in ``autolens/lens/tracer_util.py`` and
``Tracer.galaxies_ascending_redshift`` adds a JAX-aware fast-path guard that
trusts input galaxy order when any galaxy redshift is traced.

This script runs two scenarios:

- **Scenario A** — subhalo redshift fixed at z=0.55 (Python float). Exercises
  the unchanged numpy fast-path through ``tracer_util``.
- **Scenario B** — subhalo redshift is ``af.UniformPrior(0.2, 0.9)`` (becomes
  a traced scalar under ``jax.jit``). Exercises the JAX partition-and-trust-
  input-order path that fixes #498.

Both scenarios call ``fitness._vmap`` over a small batch of prior-median
parameter vectors and ``jax.jit``-wrap ``analysis.fit_from`` on a single
instance. Both must produce a finite, NumPy-matching ``log_likelihood`` and
both must produce vmap results matching the regression literals below.
A regression of #498 will trip the ``assert_allclose`` on Scenario B's vmap
output (which would either raise the original ``TracerBoolConversionError``
or silently drift away from the reference value).

Same ``dataset/imaging/jax_test`` dataset, lens, and source as ``lp.py`` —
only the ``subhalo`` galaxy is added.
"""

# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import sys

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
runs ``fitness._vmap`` over a small batch of prior-median parameter vectors
and asserts the result matches the regression literal, then jit-wraps
``analysis.fit_from`` on a single instance and asserts the jit log-likelihood
matches the NumPy-path log-likelihood within ``rtol=1e-4``. Any failure
raises (``AssertionError`` or ``TracerBoolConversionError``) and the script
exits non-zero.
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
        positions_likelihood_list=[al.PositionsLH(threshold=0.4, positions=positions)],
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

    result = fitness._vmap(parameters)
    print(
        f"  [vmap]   result shape={np.shape(np.array(result))}, "
        f"first={float(np.array(result)[0]):.6e}"
    )
    np.testing.assert_allclose(
        np.array(result),
        -1.412105e09,
        rtol=1e-4,
        err_msg="subhalo: JAX vmap likelihood mismatch (issue #498 regression?)",
    )

    # --- Path 2: jit-wrapped analysis.fit_from on a single instance ---
    instance = model.instance_from_prior_medians()

    analysis_np = al.AnalysisImaging(
        dataset=dataset,
        positions_likelihood_list=[al.PositionsLH(threshold=0.4, positions=positions)],
        use_jax=False,
    )
    fit_np = analysis_np.fit_from(instance=instance)
    print(f"  [numpy]  fit.log_likelihood = {float(fit_np.log_likelihood):.6e}")

    analysis_jit = al.AnalysisImaging(
        dataset=dataset,
        positions_likelihood_list=[al.PositionsLH(threshold=0.4, positions=positions)],
        use_jax=True,
    )
    fit = jax.jit(analysis_jit.fit_from)(instance)
    print(f"  [jit]    fit.log_likelihood = {float(fit.log_likelihood):.6e}")
    np.testing.assert_allclose(
        float(fit.log_likelihood), float(fit_np.log_likelihood), rtol=1e-4
    )
    print(f"  [jit]    matches numpy path within rtol=1e-4")


"""
__Scenario A — Fixed Subhalo Redshift__

The subhalo redshift is a Python float between the lens (z=0.5) and source
(z=1.0). Exercises the unchanged numpy fast-path through ``tracer_util``.
"""
run_scenario("A (fixed redshift z=0.55)", redshift_subhalo=0.55)


"""
__Scenario B — Free Subhalo Redshift__

The subhalo redshift is an ``af.UniformPrior(0.2, 0.9)`` — a traced scalar
under ``jax.jit``. Exercises the JAX partition-and-trust-input-order path
introduced by PyAutoLens PR #499 to fix #498. The vmap regression literal
``-1.412105e+09`` is identical to Scenario A's because both evaluate at the
prior median ``z_subhalo = 0.55``; what differs is the code path inside
``tracer_util`` (numpy sort vs JAX-aware partition).
"""
run_scenario(
    "B (free redshift UniformPrior(0.2, 0.9))",
    redshift_subhalo=af.UniformPrior(lower_limit=0.2, upper_limit=0.9),
)


print()
print("=" * 72)
print("PASS: issue #498 regression check passing")
print("=" * 72)
