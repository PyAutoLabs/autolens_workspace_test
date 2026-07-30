"""
Multi: Shared Preloads Parity
=============================

Correctness gate for the imaging multi-exposure shared-state path (PyAutoLens #600 / PyAutoArray #380,
design PyAutoLens#599) — the imaging sibling of `datacube/shared_preloads.py`.

Two identical g-band exposures are fitted via `af.FactorGraphModel` **two ways**, and the summed
log-likelihoods are asserted to match:

- `shared_preloads=False` — every exposure computes its own image-mesh and ray-traces its own
  source-plane mesh (the baseline).
- `shared_preloads=True` — the lead exposure ray-traces the source-plane mesh once and every exposure
  maps its own grid onto that identical shared Delaunay mesh.

With identical exposures the per-exposure meshes are identical to the shared one, so preloading must not
change the answer — the summed likelihood is asserted equal. (With genuinely different exposures the
shared path legitimately differs, because the lead's mesh replaces each exposure's own — that is the
feature, not an error, so exact parity is only asserted on the identical-exposure graph.)

Unlike the datacube (identical grids → whole mapper + curvature matrix shared), only the source-plane
mesh geometry is shared for imaging: each exposure builds its own mapper, PSF-blurred mapping matrix,
curvature matrix and regularization matrix. A realistic two-band (g + r) shared graph is then run under
JAX, asserting the vmap and `jit(log_likelihood_function)` round-trip agree — proving the shared path
threads `jax.jit` end-to-end.

Parity is asserted **within each backend** (numpy-vs-numpy, jax-vs-jax); see
`multi_dataset/delaunay.py` for why numpy and JAX are not compared to each other for
pixelized sources.

Run from the workspace root:

    python scripts/multi_dataset/jax_likelihood/shared_preloads.py

__Env__

Test-harness configuration (PyAutoHands docs/env_profile_redesign.md §10).
JAX likelihood functions test JIT compilation; need JAX enabled and full-
size datasets.

ENV: jax full_datasets
"""

import numpy as np
import jax
import jax.numpy as jnp
from os import path

import autofit as af
import autolens as al

pixel_scales = 0.1
mask_radius = 3.0
pixels = 500
edge_pixels_total = 30

dataset_path = path.join("dataset", "multi_dataset", "lens_sersic")

"""
__Dataset Auto-Simulation__
"""
if al.util.dataset.should_simulate(dataset_path):
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "scripts/multi_dataset/simulator.py"],
        check=True,
    )


def _dataset(band):
    dataset = al.Imaging.from_fits(
        data_path=path.join(dataset_path, f"{band}_data.fits"),
        psf_path=path.join(dataset_path, f"{band}_psf.fits"),
        noise_map_path=path.join(dataset_path, f"{band}_noise_map.fits"),
        pixel_scales=pixel_scales,
    )
    mask = al.Mask2D.circular(
        shape_native=dataset.shape_native,
        pixel_scales=dataset.pixel_scales,
        radius=mask_radius,
    )
    dataset = dataset.apply_mask(mask=mask)
    return dataset.apply_over_sampling(
        over_sample_size_lp=1, over_sample_size_pixelization=1
    )


def _adapt_images(dataset):
    galaxy_image_name_dict = {
        "('galaxies', 'lens')": dataset.data,
        "('galaxies', 'source')": dataset.data,
    }
    image_mesh = al.image_mesh.Hilbert(
        pixels=pixels, weight_power=3.5, weight_floor=0.01
    )
    image_plane_mesh_grid = image_mesh.image_plane_mesh_grid_from(
        mask=dataset.mask,
        adapt_data=galaxy_image_name_dict["('galaxies', 'source')"],
    )
    image_plane_mesh_grid = al.image_mesh.append_with_circle_edge_points(
        image_plane_mesh_grid=image_plane_mesh_grid,
        centre=dataset.mask.mask_centre,
        radius=mask_radius + dataset.mask.pixel_scale / 2.0,
        n_points=edge_pixels_total,
    )
    return al.AdaptImages(
        galaxy_name_image_dict=galaxy_image_name_dict,
        galaxy_name_image_plane_mesh_grid_dict={
            "('galaxies', 'source')": image_plane_mesh_grid
        },
    )


def _model():
    mass = af.Model(al.mp.Isothermal)
    mass.centre.centre_0 = af.UniformPrior(lower_limit=-0.05, upper_limit=0.05)
    mass.centre.centre_1 = af.UniformPrior(lower_limit=-0.05, upper_limit=0.05)
    mass.einstein_radius = af.UniformPrior(lower_limit=1.55, upper_limit=1.65)
    mass.ell_comps.ell_comps_0 = af.UniformPrior(lower_limit=-0.01, upper_limit=0.01)
    mass.ell_comps.ell_comps_1 = af.UniformPrior(lower_limit=0.045, upper_limit=0.060)

    lens = af.Model(al.Galaxy, redshift=0.5, mass=mass)

    pixelization = af.Model(
        al.Pixelization,
        mesh=al.mesh.Delaunay(pixels=pixels, zeroed_pixels=edge_pixels_total),
        regularization=al.reg.ConstantSplit,
    )
    source = af.Model(al.Galaxy, redshift=1.0, pixelization=pixelization)

    return af.Collection(galaxies=af.Collection(lens=lens, source=source))


def _factor_graph(band_list, shared_preloads, use_jax):
    model = _model()

    analysis_list = []
    for band in band_list:
        dataset = _dataset(band)
        analysis_list.append(
            al.AnalysisImaging(
                dataset=dataset,
                adapt_images=_adapt_images(dataset),
                use_jax=use_jax,
                shared_preloads=shared_preloads,
                raise_inversion_positions_likelihood_exception=False,
            )
        )

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


def _assert_identical_exposure_parity(use_jax):
    """Two identical g-band exposures: the shared mesh equals each exposure's own mesh, so
    shared-vs-unshared must agree exactly."""
    backend = "JAX" if use_jax else "numpy"

    ll_unshared = _log_likelihood(_factor_graph(["g", "g"], False, use_jax), use_jax)
    ll_shared = _log_likelihood(_factor_graph(["g", "g"], True, use_jax), use_jax)

    print(
        f"[{backend}] 2x g-band log likelihood  unshared={ll_unshared}  shared={ll_shared}"
    )

    np.testing.assert_allclose(
        ll_shared,
        ll_unshared,
        rtol=1e-7,
        err_msg=(
            f"multi_dataset/shared_preloads ({backend}): shared_preloads=True changed the summed "
            f"log-likelihood on identical exposures. Sharing the exposure-invariant "
            f"source-plane mesh must be exact."
        ),
    )


def _assert_two_band_shared_jit():
    """Realistic g + r graph under the shared mesh: vmap and jit round-trip must agree,
    proving the shared path (lead-factor mesh trace + per-exposure mapping) is jit-safe.
    """
    from autofit.non_linear.fitness import Fitness

    factor_graph = _factor_graph(["g", "r"], True, use_jax=True)

    fitness = Fitness(
        model=factor_graph.global_prior_model,
        analysis=factor_graph,
        fom_is_log_likelihood=True,
        resample_figure_of_merit=-1.0e99,
    )

    medians = factor_graph.global_prior_model.physical_values_from_prior_medians
    parameters = jnp.array(np.tile(np.asarray(medians), (2, 1)))
    vmap_result = np.array(fitness._vmap(parameters))
    print(f"[JAX] g+r shared-mesh vmap log likelihood: {vmap_result}")
    assert np.all(np.isfinite(vmap_result)), "shared-mesh vmap likelihood not finite"

    @jax.jit
    def log_l_jit_fn(params):
        instance = factor_graph.global_prior_model.instance_from_vector(
            vector=params, xp=jnp
        )
        return factor_graph.log_likelihood_function(instance)

    log_l_jit = float(log_l_jit_fn(jnp.array(medians)))
    print(f"[JAX] g+r shared-mesh jit log likelihood:  {log_l_jit}")

    np.testing.assert_allclose(
        log_l_jit,
        vmap_result[0],
        rtol=1e-6,
        err_msg="multi_dataset/shared_preloads: shared-mesh jit round-trip disagrees with vmap",
    )


if __name__ == "__main__":
    _assert_identical_exposure_parity(use_jax=False)
    _assert_identical_exposure_parity(use_jax=True)
    _assert_two_band_shared_jit()
    print("shared_preloads: multi shared-vs-unshared parity + shared jit all passed")
