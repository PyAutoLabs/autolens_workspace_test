"""
Datacube: Shared Preloads Parity
================================

Correctness gate for the datacube cross-`Analysis` shared-state path (PyAutoLens #566 / PyAutoArray #344).

An identical-channel datacube is fitted via `af.FactorGraphModel` **two ways**, and the per-cube
log-likelihoods are asserted to match:

- `shared_preloads=False` — every channel rebuilds its own inversion-setup quantities (the baseline).
- `shared_preloads=True` — the channel-invariant quantities (the source-plane mapper and the curvature
  matrix `F = LᵀW̃L`) are computed once on the lead factor and reused by every channel.

Preloading must not change the answer: the whole point is that the shared quantities are *identical* to
the per-channel ones, so the summed likelihood is bit-for-bit the same. This script builds both paths
itself (independent of the other datacube scripts) so the with/without coverage is explicit.

The sparse (w-tilde) interferometer route is used (`apply_sparse_operator`) — the production datacube
path — so both short-circuits are exercised: the mapper reuse in `TracerToInversion` and the curvature
reuse in `InversionInterferometerSparse`.

Parity is asserted **within each backend** (numpy-vs-numpy, jax-vs-jax). numpy and JAX are not compared
to each other: for pixelized sources `FactorGraphModel.log_likelihood_function` takes a different
numerical path under `use_jax=True` than under `use_jax=False` (see
`interferometer/rectangular_datacube.py`).

Run from the workspace root:

    python scripts/interferometer/shared_preloads_datacube.py

__Env__

Test-harness configuration (PyAutoHands docs/env_profile_redesign.md §10).
JAX likelihood functions test JIT compilation; need JAX enabled and full-
size datasets.

ENV: jax full_datasets
"""

import numpy as np
import jax.numpy as jnp
from os import path

import autofit as af
import autolens as al

n_channels = 3

real_space_mask = al.Mask2D.circular(
    shape_native=(128, 128),
    pixel_scales=0.2,
    radius=3.0,
)

dataset_name = "simple"
dataset_path = path.join("dataset", "interferometer", dataset_name)

if al.util.dataset.should_simulate(dataset_path):
    import subprocess
    import sys

    subprocess.run(
        [
            sys.executable,
            "scripts/interferometer/simulator/simple.py",
        ],
        check=True,
    )


def _model():
    # Uniform priors centred on the values used by `scripts/interferometer/simulator/simple.py`:
    # an `Isothermal` with an axis ratio of 0.9 at 45 degrees and an Einstein radius of 1.6, plus
    # an external shear of (0.05, 0.05) — the shear the data contains and the model must carry.
    mass_ell_comps = al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0)

    mass = af.Model(al.mp.Isothermal)
    mass.centre.centre_0 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
    mass.centre.centre_1 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
    mass.einstein_radius = af.UniformPrior(lower_limit=1.5, upper_limit=1.7)
    mass.ell_comps.ell_comps_0 = af.UniformPrior(
        lower_limit=mass_ell_comps[0] - 0.05, upper_limit=mass_ell_comps[0] + 0.05
    )
    mass.ell_comps.ell_comps_1 = af.UniformPrior(
        lower_limit=mass_ell_comps[1] - 0.05, upper_limit=mass_ell_comps[1] + 0.05
    )

    shear = af.Model(al.mp.ExternalShear)
    shear.gamma_1 = af.UniformPrior(lower_limit=0.04, upper_limit=0.06)
    shear.gamma_2 = af.UniformPrior(lower_limit=0.04, upper_limit=0.06)

    lens = af.Model(al.Galaxy, redshift=0.5, mass=mass, shear=shear)

    pixelization = al.Pixelization(
        mesh=al.mesh.RectangularUniform(shape=(8, 8)),
        regularization=al.reg.Constant(coefficient=1.0),
    )
    source = af.Model(al.Galaxy, redshift=1.0, pixelization=pixelization)

    return af.Collection(galaxies=af.Collection(lens=lens, source=source))


def _factor_graph(shared_preloads, use_jax):
    dataset_list = [
        al.Interferometer.from_fits(
            data_path=path.join(dataset_path, "data.fits"),
            noise_map_path=path.join(dataset_path, "noise_map.fits"),
            uv_wavelengths_path=path.join(dataset_path, "uv_wavelengths.fits"),
            real_space_mask=real_space_mask,
            transformer_class=al.TransformerDFT,
        ).apply_sparse_operator(use_jax=use_jax)
        for _ in range(n_channels)
    ]

    model = _model()

    analysis_list = [
        al.AnalysisInterferometer(
            dataset=dataset,
            use_jax=use_jax,
            shared_preloads=shared_preloads,
            raise_inversion_positions_likelihood_exception=False,
        )
        for dataset in dataset_list
    ]

    analysis_factor_list = [
        af.AnalysisFactor(prior_model=model.copy(), analysis=analysis)
        for analysis in analysis_list
    ]

    return af.FactorGraphModel(*analysis_factor_list, use_jax=use_jax)


def _log_likelihood(factor_graph, use_jax):
    xp = jnp if use_jax else np
    params = factor_graph.global_prior_model.physical_values_from_prior_medians
    vector = jnp.array(params) if use_jax else params
    instance = factor_graph.global_prior_model.instance_from_vector(
        vector=vector, xp=xp
    )
    return float(factor_graph.log_likelihood_function(instance))


def _assert_parity(use_jax):
    backend = "JAX" if use_jax else "numpy"

    ll_unshared = _log_likelihood(_factor_graph(False, use_jax), use_jax)
    ll_shared = _log_likelihood(_factor_graph(True, use_jax), use_jax)

    print(
        f"[{backend}] cube log likelihood  unshared={ll_unshared}  shared={ll_shared}"
    )

    np.testing.assert_allclose(
        ll_shared,
        ll_unshared,
        rtol=1e-7,
        err_msg=(
            f"datacube/shared_preloads ({backend}): shared_preloads=True changed the cube "
            f"log-likelihood. Preloading the channel-invariant mapper / curvature must be exact."
        ),
    )


if __name__ == "__main__":
    _assert_parity(use_jax=False)
    _assert_parity(use_jax=True)
    print("shared_preloads: numpy and JAX shared-vs-unshared parity all passed")
