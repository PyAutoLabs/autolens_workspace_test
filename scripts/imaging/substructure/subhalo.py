"""
JAX Regression: Free Subhalo Redshift + NFWMCRLudlow Under jax.jit
==================================================================

This script bundles two JAX regression checks that both live in the
subhalo-likelihood code path:

1. **Free subhalo redshift** (issue
   https://github.com/PyAutoLabs/PyAutoLens/issues/498, fixed in
   PyAutoLens PR #499). When the bug was present, setting
   ``subhalo.redshift = af.UniformPrior(...)`` raised
   ``jax.errors.TracerBoolConversionError`` because Python ``sorted`` /
   ``float()`` / ``<=`` / ``==`` were called on what became a traced
   scalar under ``jax.jit``. The fix in ``autolens/lens/tracer_util.py``
   and ``Tracer.galaxies_ascending_redshift`` adds a JAX-aware fast-path
   guard that trusts input galaxy order when any galaxy redshift is
   traced.

2. **NFWMCRLudlow subhalo under JAX** (issue
   https://github.com/PyAutoLabs/PyAutoGalaxy/issues/397 and follow-up
   #403). ``mp.NFWMCRLudlowSph`` calls into ``colossus.halo.concentration``
   via ``jax.pure_callback`` in ``mcr_util.py`` to compute the
   mass-concentration relation. We need a regression check that this
   ``pure_callback`` chain co-operates with ``fitness._vmap`` (the path
   that actually exercises the callback inside a JAX trace — single-call
   ``jax.jit(analysis.fit_from)(instance)`` receives a pre-built model
   instance whose ``kappa_s`` was already computed at construction time,
   so it does *not* exercise the callback under jit).

Four scenarios run, all on the same ``jax_test`` imaging dataset (lens
at z=0.5, source at z=1.0):

| Scenario | Subhalo mass    | Subhalo redshift                 |
|----------|-----------------|----------------------------------|
| A        | ``IsothermalSph``  | fixed at z=0.55                  |
| B        | ``IsothermalSph``  | free ``UniformPrior(0.2, 0.9)``  |
| C        | ``NFWMCRLudlowSph``| fixed at z=0.55                  |
| D        | ``NFWMCRLudlowSph``| free ``UniformPrior(0.2, 0.9)``  |

Each scenario calls ``fitness._vmap`` over a small batch of prior-median
parameter vectors and ``jax.jit``-wraps ``analysis.fit_from`` on a single
instance. The vmap result is asserted against a hard-coded regression
literal; the single-instance ``jit`` log-likelihood is asserted to match
the NumPy-path log-likelihood within ``rtol=1e-4``. Any failure raises
(``AssertionError`` or ``TracerBoolConversionError``) and the script
exits non-zero.

A regression of issue #498 will trip Scenario B's vmap assert. A
regression of the Ludlow pure_callback path will trip Scenario C or D's
vmap assert (or raise inside the callback under vmap).

__Env__

Test-harness configuration (PyAutoHands docs/env_profile_redesign.md §10).
JAX likelihood functions test JIT compilation; need JAX enabled and full-
size datasets.

ENV: jax full_datasets
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
        [sys.executable, "scripts/imaging/simulator/simple.py"],
        check=True,
    )

dataset = al.Imaging.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    pixel_scales=0.3,
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
__Subhalo Mass Factories__

Each factory builds an ``af.Model`` for the subhalo mass profile. The
factory receives the subhalo's galaxy redshift so that
``NFWMCRLudlowSph.redshift_object`` can be tied to it (matching the
physical setup where the mass-concentration relation is evaluated at
the subhalo's own redshift). For the Isothermal factory the argument
is unused — kept for a uniform factory signature.
"""


def isothermal_subhalo_mass(redshift_subhalo):
    m = af.Model(al.mp.IsothermalSph)
    m.centre_0 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
    m.centre_1 = af.UniformPrior(lower_limit=1.2, upper_limit=1.8)
    m.einstein_radius = af.UniformPrior(lower_limit=0.01, upper_limit=0.4)
    return m


def nfw_mcr_ludlow_subhalo_mass(redshift_subhalo):
    m = af.Model(al.mp.NFWMCRLudlowSph)
    m.centre_0 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
    m.centre_1 = af.UniformPrior(lower_limit=1.2, upper_limit=1.8)
    m.mass_at_200 = af.LogUniformPrior(lower_limit=1.0e8, upper_limit=1.0e11)
    m.redshift_object = redshift_subhalo
    m.redshift_source = 1.0
    return m


"""
__Model Builder__

A helper that builds the full ``af.Collection`` for a given subhalo
redshift and mass-profile factory. The lens and source pieces match
``lp.py``. The galaxy attribute is deliberately named ``subhalo`` so
that ``AnalysisLens.tracer_via_instance_from`` enters the relevant
branch at ``analysis/lens.py:105`` (``hasattr(instance.galaxies,
"subhalo")``).
"""


def build_model(redshift_subhalo, subhalo_mass_factory):
    lens = af.Model(
        al.Galaxy,
        redshift=0.5,
        bulge=af.Model(al.lp_linear.Sersic),
        mass=af.Model(al.mp.PowerLaw),
        shear=af.Model(al.mp.ExternalShear),
    )

    subhalo_mass = subhalo_mass_factory(redshift_subhalo)

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

Builds an ``AnalysisImaging`` (with positions likelihood, mirroring
``lp.py``), runs ``fitness._vmap`` over a small batch of prior-median
parameter vectors and asserts the result matches the regression literal,
then jit-wraps ``analysis.fit_from`` on a single instance and asserts
the jit log-likelihood matches the NumPy-path log-likelihood within
``rtol=1e-4``.

If ``expected_vmap`` is ``None`` the script is in calibration mode: it
prints the computed value and skips the assert so the literal can be
copied back into the script.
"""

from autofit.non_linear.fitness import Fitness


def run_scenario(
    label,
    redshift_subhalo,
    subhalo_mass_factory,
    expected_vmap,
    batch_size=4,
):
    print()
    print("=" * 72)
    print(
        f"Scenario {label}: redshift_subhalo={redshift_subhalo!r}, "
        f"mass={subhalo_mass_factory.__name__}"
    )
    print("=" * 72)

    model = build_model(redshift_subhalo, subhalo_mass_factory)

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
    first = float(np.array(result)[0])
    print(
        f"  [vmap]   result shape={np.shape(np.array(result))}, " f"first={first:.6e}"
    )
    if expected_vmap is None:
        print(
            f"  [vmap]   CALIBRATION — paste expected_vmap={first:.6e} into the script"
        )
    else:
        np.testing.assert_allclose(
            np.array(result),
            expected_vmap,
            rtol=1e-4,
            err_msg=f"subhalo[{label}]: JAX vmap likelihood mismatch",
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
__Scenario A — IsothermalSph, Fixed Subhalo Redshift__

The subhalo redshift is a Python float between the lens (z=0.5) and source
(z=1.0). Exercises the unchanged numpy fast-path through ``tracer_util``.
"""
run_scenario(
    "A (Isothermal, fixed redshift z=0.55)",
    redshift_subhalo=0.55,
    subhalo_mass_factory=isothermal_subhalo_mass,
    expected_vmap=-7.062287e08,
)


"""
__Scenario B — IsothermalSph, Free Subhalo Redshift__

The subhalo redshift is an ``af.UniformPrior(0.2, 0.9)`` — a traced scalar
under ``jax.jit``. Exercises the JAX partition-and-trust-input-order path
introduced by PyAutoLens PR #499 to fix #498. The vmap regression literal
is identical to Scenario A's because both evaluate at the prior median
``z_subhalo = 0.55``; what differs is the code path inside ``tracer_util``
(numpy sort vs JAX-aware partition).
"""
run_scenario(
    "B (Isothermal, free redshift UniformPrior(0.2, 0.9))",
    redshift_subhalo=af.UniformPrior(lower_limit=0.2, upper_limit=0.9),
    subhalo_mass_factory=isothermal_subhalo_mass,
    expected_vmap=-7.062287e08,
)


"""
__Scenario C — NFWMCRLudlowSph, Fixed Subhalo Redshift__

The subhalo mass profile is ``mp.NFWMCRLudlowSph``, which calls into
``colossus.halo.concentration`` via ``jax.pure_callback`` (in
``mcr_util.py``) to compute the mass-concentration relation. Under
``fitness._vmap`` the callback fires once per batch element (because
``vmap_method="sequential"``), so this scenario is the load-bearing
regression check for the pure_callback path under JAX.

The single-instance ``jit`` path further down does NOT exercise the
callback under jit — the model instance is built outside the trace
with concrete inputs, so ``kappa_s`` is already a concrete float by
the time ``analysis.fit_from`` is jit-compiled.
"""
run_scenario(
    "C (NFWMCRLudlow, fixed redshift z=0.55)",
    redshift_subhalo=0.55,
    subhalo_mass_factory=nfw_mcr_ludlow_subhalo_mass,
    expected_vmap=-6.747760e08,
)


"""
__Scenario D — NFWMCRLudlowSph, Free Subhalo Redshift__

Combines the issue #498 fix (free traced subhalo redshift) with the
NFWMCRLudlow pure_callback chain. The subhalo galaxy redshift and the
mass profile's ``redshift_object`` are tied (via the factory) so the
mass-concentration relation is evaluated at the subhalo's own redshift,
which becomes a traced scalar.
"""
run_scenario(
    "D (NFWMCRLudlow, free redshift UniformPrior(0.2, 0.9))",
    redshift_subhalo=af.UniformPrior(lower_limit=0.2, upper_limit=0.9),
    subhalo_mass_factory=nfw_mcr_ludlow_subhalo_mass,
    expected_vmap=-6.747760e08,
)


print()
print("=" * 72)
print("PASS: subhalo regression checks passing")
print("=" * 72)
