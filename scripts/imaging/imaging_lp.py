"""
Tests JAX gradients of the imaging log-likelihood for parametric light-profile
models, in two stages:

 1. Finiteness — ``jax.value_and_grad`` returns a finite log-likelihood and a
    finite, non-zero gradient vector (the original check).
 2. Correctness — the autodiff gradient agrees with central finite differences
    parameter-by-parameter (see ``util.py``).

Two model variants are exercised so both light-profile evaluation paths are
covered:

 - ``lp.Sersic`` — standard profiles with free ``intensity`` (pure profile
   evaluation, no inversion).
 - ``lp_linear.Sersic`` — linear profiles whose intensities are solved by the
   linear inversion (positive-only NNLS solve inside the likelihood).

Autodiff runs on the eager (un-jitted) likelihood; the finite-difference
evaluations use a jitted likelihood for speed, guarded by an eager-vs-jit
consistency check at the base point so ``pure_callback`` constant-folding
cannot fake the comparison.

__Env__

Test-harness configuration (PyAutoHands docs/env_profile_redesign.md §10).
Drive jax.value_and_grad + finite-difference gradient checks; need JAX
enabled and full-resolution float64 data.

ENV: jax full_datasets
"""

import numpy as np
import jax
import jax.numpy as jnp
from os import path

import autofit as af
import autolens as al

import os
import sys

# ``util.py`` lives in ``scripts/misc/`` after the mirror restructure; add it to
# the path so this script (run as a file, cwd = workspace root) can import it.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "misc"))

import util

dataset_name = "jax_test"
dataset_path = path.join("dataset", "imaging", dataset_name)

"""
__Dataset Auto-Simulation__

If the dataset does not already exist on your system, it will be created by running the corresponding
simulator script. This ensures that all example scripts can be run without manually simulating data first.
"""
if al.util.dataset.should_simulate(dataset_path):
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "scripts/imaging/simulator.py"],
        check=True,
    )

dataset = al.Imaging.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    pixel_scales=0.2,
)

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


def model_from(bulge_cls):
    """
    The lens + source model used by both variants; only the bulge light-profile
    class differs (standard vs linear Sersic).
    """
    lens = af.Model(
        al.Galaxy,
        redshift=0.5,
        bulge=af.Model(bulge_cls),
        mass=af.Model(al.mp.PowerLaw),
        shear=af.Model(al.mp.ExternalShear),
    )
    source = af.Model(al.Galaxy, redshift=1.0, bulge=af.Model(bulge_cls))
    return af.Collection(galaxies=af.Collection(lens=lens, source=source))


from autofit.non_linear.fitness import Fitness

for variant, bulge_cls in [
    ("lp.Sersic (standard)", al.lp.Sersic),
    ("lp_linear.Sersic (linear)", al.lp_linear.Sersic),
]:
    print(f"\n=== {variant} ===")

    model = model_from(bulge_cls=bulge_cls)

    analysis = al.AnalysisImaging(
        dataset=dataset,
        positions_likelihood_list=[al.PositionsLH(threshold=0.4, positions=positions)],
    )

    fitness = Fitness(
        model=model,
        analysis=analysis,
        fom_is_log_likelihood=True,
        resample_figure_of_merit=-1.0e99,
    )

    param_vector = jnp.array(model.physical_values_from_prior_medians)

    # Perturb ell_comps away from (0,0) to avoid degenerate gradients at the
    # circular-profile singularity (arctan2 gradient is undefined at exactly (0,0)).
    key = jax.random.PRNGKey(0)
    perturbation = jax.random.uniform(
        key, shape=param_vector.shape, minval=0.01, maxval=0.05
    )
    param_vector = param_vector + perturbation

    # Anchor the mass model and source near the simulator truth (jax_test). At raw
    # prior medians the fit is so poor that the positive-only NNLS solves the linear
    # source intensity to zero, which makes every source-shape gradient legitimately
    # zero — the comparison would then never exercise gradient flow through the
    # source. Values mirror scripts/imaging/simulator.py.
    overrides = {
        # The lens bulge's prior-median effective_radius is ~15", which floods the
        # whole 3" mask and lets the (linear) bulge absorb the source arcs — NNLS
        # then zeroes the source amplitude. Pin the bulge to truth-scale values.
        "galaxies.lens.bulge.effective_radius": 0.6,
        "galaxies.lens.bulge.sersic_index": 3.0,
        "galaxies.lens.mass.centre_0": 0.01,
        "galaxies.lens.mass.centre_1": 0.01,
        "galaxies.lens.mass.ell_comps_0": al.convert.ell_comps_from(
            axis_ratio=0.8, angle=45.0
        )[0],
        "galaxies.lens.mass.ell_comps_1": al.convert.ell_comps_from(
            axis_ratio=0.8, angle=45.0
        )[1],
        "galaxies.lens.mass.einstein_radius": 1.6,
        "galaxies.lens.mass.slope": 2.0,
        "galaxies.source.bulge.centre_0": 0.1,
        "galaxies.source.bulge.centre_1": 0.1,
        "galaxies.source.bulge.ell_comps_0": al.convert.ell_comps_from(
            axis_ratio=0.8, angle=60.0
        )[0],
        "galaxies.source.bulge.ell_comps_1": al.convert.ell_comps_from(
            axis_ratio=0.8, angle=60.0
        )[1],
        "galaxies.source.bulge.effective_radius": 1.0,
        "galaxies.source.bulge.sersic_index": 1.0,
    }
    param_names = util.parameter_names_from(model)
    param_vector = np.array(param_vector)
    for name, value in overrides.items():
        if name in param_names:
            param_vector[param_names.index(name)] = value
    param_vector = jnp.array(param_vector)

    value, grad = jax.value_and_grad(fitness.call)(param_vector)

    print(f"Log likelihood = {float(value):.6f}")
    print(f"Gradient shape = {grad.shape}")

    assert np.isfinite(float(value)), "Log likelihood is not finite"
    assert grad.shape == (
        model.total_free_parameters,
    ), f"Gradient shape mismatch: {grad.shape}"
    assert np.all(
        np.isfinite(np.array(grad))
    ), f"Gradient contains non-finite values: {np.array(grad)}"
    assert not np.all(np.array(grad) == 0.0), "Gradient is all zeros"

    """
    __Finite-Difference Correctness__
    """
    f_jit = jax.jit(fitness.call)

    util.assert_eager_jit_consistent(fitness.call, f_jit, param_vector)

    comparison = util.compare_gradients(
        fitness.call,
        param_vector,
        param_names=param_names,
        f_fd=f_jit,
    )

    util.assert_gradients_match(comparison)

    # Guard against a silently dead source component (see the anchoring comment
    # above): the source-shape gradients must actually be non-zero here.
    source_indices = [
        i for i, name in enumerate(param_names) if "galaxies.source" in name
    ]
    assert np.any(
        np.abs(comparison["ad"][source_indices]) > 1e-3
    ), "All source-parameter gradients are ~zero — NNLS zeroed the source; move the evaluation point."

    print(f"{variant}: autodiff matches finite differences.")

print("\nimaging_lp.py JAX gradient checks passed.")
