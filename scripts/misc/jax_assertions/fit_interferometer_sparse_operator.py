"""
Jax Assertions: FitInterferometer Sparse Operator
=================================================

Cross-implementation parity check for the ``InterferometerSparseOperator`` path
inside a ``FitInterferometer`` fit. This is the interferometer counterpart of
``fit_imaging_sparse_operator.py`` in this directory.

A tracer whose linear objects are a MIXED list — a linear light profile func
list (``lp_linear.Sersic``) plus one or more ``Mapper`` objects — is fitted
twice:

- via the standard mapping-matrix path (``InversionInterferometerMapping``), and
- via ``dataset.apply_sparse_operator()`` (``InversionInterferometerSparse``).

The two must agree on ``curvature_matrix``, ``data_vector``,
``regularization_matrix``, ``reconstruction`` and ``log_evidence``.

This guards PyAutoArray #499 / #500. Before that fix
``InversionInterferometerSparse.curvature_matrix`` returned only the
single-mapper diagonal block, yet the factory still routed a mixed
``[AbstractLinearObjFuncList, Mapper]`` list (or several mappers) to the sparse
class whenever a sparse operator was attached — the linear-function and
cross-mapper terms were silently dropped from F. The func-list block, the
func-list/mapper off-diagonals and the mapper/mapper off-diagonals are all
exercised below, so a regression to the old behaviour fails here.

__Noise-map choice__

The noise-map is deliberately NON-uniform across visibilities: a unit noise-map
makes the inverse-variance weighting the identity and would hide any weighting
bug in the sparse operator's ``W~ = Re(F^H W F)``.

The real and imaginary sigma of each visibility are kept EQUAL, which is what
``SimulatorInterferometer`` produces and what real datasets carry. That
equality is a precondition of the ``Re(F^H W F)`` reduction itself: with
``sigma_real != sigma_imag`` the curvature is
``sum_k [Re(F_ki)Re(F_kj)/sr^2 + Im(F_ki)Im(F_kj)/si^2]``, which does not
collapse to a single real operator. Feeding unequal real/imag sigmas here makes
the sparse and mapping paths disagree by ~3e-2 relative even on the
single-mapper path that #500 left untouched — a pre-existing limitation of the
formalism, not something this script is written to gate.

__Env__

Test-harness configuration (PyAutoHands docs/env_profile_redesign.md §10).
The sparse operator is the JAX accelerator path this script exists to gate, so
JAX must stay enabled. The dataset is already tiny and built in-memory, so the
SMALL_DATASETS cap is irrelevant either way.

ENV: jax
"""

import numpy as np

import autoarray as aa
import autolens as al


"""
__Dataset__

A small in-memory `Interferometer`: a 15 x 15 real-space mask (109 unmasked
pixels) and 150 random baselines. The uv range is chosen so `1 / uv` sits
inside the mask (`1e6` wavelengths ~ 0.2"); shorter baselines leave the source
unresolved and the inversion degenerate.

`TransformerDFT` is used throughout — at `N_vis * N_pix ~ 1.6e4` it is far
below the ~1e7 NUFFT crossover.
"""
rng = np.random.default_rng(1234)

real_space_mask = al.Mask2D.circular(
    shape_native=(15, 15),
    pixel_scales=0.1,
    radius=0.6,
)

grid = al.Grid2D.from_mask(mask=real_space_mask)

uv_wavelengths = rng.uniform(-1.0e6, 1.0e6, size=(150, 2))

simulator = al.SimulatorInterferometer(
    uv_wavelengths=uv_wavelengths,
    exposure_time=100.0,
    transformer_class=al.TransformerDFT,
    noise_sigma=0.1,
    noise_seed=1,
)

tracer_simulate = al.Tracer(
    galaxies=[
        al.Galaxy(
            redshift=0.5,
            light=al.lp.Sersic(centre=(0.0, 0.0), effective_radius=0.2),
            mass=al.mp.Isothermal(centre=(0.0, 0.0), einstein_radius=0.3),
        ),
        al.Galaxy(
            redshift=1.0,
            bulge=al.lp.Sersic(centre=(0.05, 0.05), effective_radius=0.1),
        ),
    ]
)

dataset_simulated = simulator.via_tracer_from(tracer=tracer_simulate, grid=grid)

"""
The simulator writes a constant noise-map; replace it with a per-visibility one
(same sigma for the real and imaginary part, see the module docstring).
"""
sigma = rng.uniform(0.5, 2.0, size=(uv_wavelengths.shape[0], 1))

dataset = al.Interferometer(
    data=dataset_simulated.data,
    noise_map=al.VisibilitiesNoiseMap(np.repeat(sigma, 2, axis=1)),
    uv_wavelengths=uv_wavelengths,
    real_space_mask=real_space_mask,
    transformer_class=al.TransformerDFT,
)

"""
__Sparse Operator__

`use_jax=False` builds the operator eagerly; at this size that is faster than
the JAX path and the resulting `Interferometer` routes to
`InversionInterferometerSparse` either way (asserted below).
"""
dataset_sparse_operator = dataset.apply_sparse_operator(use_jax=False)


def pixelization(shape):
    return al.Pixelization(
        mesh=al.mesh.RectangularUniform(shape=shape),
        regularization=al.reg.Constant(coefficient=1.0),
    )


lens = al.Galaxy(
    redshift=0.5,
    light=al.lp_linear.Sersic(centre=(0.0, 0.0), effective_radius=0.2),
    mass=al.mp.Isothermal(centre=(0.0, 0.0), einstein_radius=0.3),
)

"""
The positive-only (NNLS) solver is the production default, but on a fit this
small its active set clamps all but one source pixel to exactly `0.0`, which
would make the `reconstruction` assertion vacuous. The direct solve is used so
every reconstructed value is non-trivial; the curvature matrix and data vector
that feed it are solver-independent.
"""
settings = al.Settings(use_positive_only_solver=False)


def assert_parity(name, tracer, rtol_reconstruction, rtol_log_evidence):
    fit_mapping = al.FitInterferometer(
        dataset=dataset,
        tracer=tracer,
        settings=settings,
    )

    fit_sparse_operator = al.FitInterferometer(
        dataset=dataset_sparse_operator,
        tracer=tracer,
        settings=settings,
    )

    inversion_mapping = fit_mapping.inversion
    inversion_sparse_operator = fit_sparse_operator.inversion

    assert isinstance(inversion_mapping, aa.InversionInterferometerMapping), (
        f"{name}: expected the un-operated dataset to route to "
        f"InversionInterferometerMapping, got {type(inversion_mapping).__name__}"
    )
    assert isinstance(inversion_sparse_operator, aa.InversionInterferometerSparse), (
        f"{name}: expected the sparse-operator dataset to route to "
        f"InversionInterferometerSparse, got "
        f"{type(inversion_sparse_operator).__name__}"
    )

    """
    The mixed linear-object list is what #499 regressed on — assert it is
    actually mixed, so a future default change cannot quietly turn this into a
    mapper-only fit that passes without testing anything new.
    """
    linear_obj_types = [type(obj).__name__ for obj in inversion_mapping.linear_obj_list]
    assert any(
        "FuncList" in linear_obj_type for linear_obj_type in linear_obj_types
    ), f"{name}: no linear func list in {linear_obj_types}"
    assert (
        sum("Mapper" in linear_obj_type for linear_obj_type in linear_obj_types) >= 1
    ), f"{name}: no mapper in {linear_obj_types}"

    """
    Shape first: before #500 the sparse `curvature_matrix` was the single-mapper
    diagonal block ALONE, so it comes back smaller than the mapping path's F and
    a value comparison would die on a broadcast error rather than say what is
    wrong. Verified against PyAutoArray 86e2944a (the commit before #500): this
    assertion is what fires there, reporting (9, 9) against (10, 10).
    """
    for quantity in ("curvature_matrix", "data_vector", "reconstruction"):
        shape_sparse_operator = np.asarray(
            getattr(inversion_sparse_operator, quantity)
        ).shape
        shape_mapping = np.asarray(getattr(inversion_mapping, quantity)).shape
        assert shape_sparse_operator == shape_mapping, (
            f"{name}: sparse-operator {quantity} has shape "
            f"{shape_sparse_operator}, mapping path has {shape_mapping} — the "
            f"sparse path is not forming every linear-object block"
        )

    """
    Every quantity is compared as a maximum ABSOLUTE difference normalised by
    the largest entry of the mapping-path array — a plain elementwise relative
    error is meaningless on a curvature matrix whose entries span nine orders
    of magnitude. The measured values are printed so drift is visible in the
    smoke log rather than only on failure.
    """

    def max_relative_error(sparse_operator, mapping):
        sparse_operator = np.asarray(sparse_operator)
        mapping = np.asarray(mapping)
        scale = float(np.abs(mapping).max())
        scale = scale if scale > 0.0 else 1.0
        return float(np.abs(sparse_operator - mapping).max()) / scale

    error_dict = {
        "curvature_matrix": max_relative_error(
            inversion_sparse_operator.curvature_matrix,
            inversion_mapping.curvature_matrix,
        ),
        "data_vector": max_relative_error(
            inversion_sparse_operator.data_vector,
            inversion_mapping.data_vector,
        ),
        "regularization_matrix": max_relative_error(
            inversion_sparse_operator.regularization_matrix,
            inversion_mapping.regularization_matrix,
        ),
        "reconstruction": max_relative_error(
            inversion_sparse_operator.reconstruction,
            inversion_mapping.reconstruction,
        ),
        "log_evidence": abs(
            float(fit_sparse_operator.log_evidence) - float(fit_mapping.log_evidence)
        )
        / abs(float(fit_mapping.log_evidence)),
    }

    for quantity, error in error_dict.items():
        print(f"  {name} | {quantity}: max relative error {error:.2e}")

    """
    `curvature_matrix`, `data_vector` and `regularization_matrix` are the
    quantities #500 changed. They are formed by direct summation (the
    regularization matrix does not touch the dataset at all), so all three
    agree at float64 round-off in every block; `rtol` is fixed at 1e-12 here,
    which still fails hard on the old behaviour where whole blocks were zero.
    """
    for quantity in ("curvature_matrix", "data_vector", "regularization_matrix"):
        assert error_dict[quantity] < 1.0e-12, (
            f"{name}: {quantity} max relative error {error_dict[quantity]:.3e} "
            f"exceeds 1e-12"
        )

    """
    `reconstruction` and `log_evidence` solve/invert `F + H` and therefore
    inherit that system's conditioning; their tolerances are set per block by
    the caller (see the call sites below).
    """
    assert error_dict["reconstruction"] < rtol_reconstruction, (
        f"{name}: reconstruction max relative error "
        f"{error_dict['reconstruction']:.3e} exceeds {rtol_reconstruction:.1e}"
    )

    assert error_dict["log_evidence"] < rtol_log_evidence, (
        f"{name}: log_evidence relative error {error_dict['log_evidence']:.3e} "
        f"exceeds {rtol_log_evidence:.1e}"
    )


"""
__Linear Light + One Mapper__

The minimal mixed list: a `LightProfileLinearObjFuncList` and a single
`Mapper`. Exercises the func-list diagonal block and the func-list/mapper
off-diagonals, neither of which the pre-#500 sparse path formed.

`F + H` here has condition number ~8e5, so the solve is well conditioned and
the reconstruction and log evidence agree at round-off — measured 3.4e-15 and
4.3e-16 respectively, against the 1e-11 / 1e-12 gates set below.
"""
assert_parity(
    name="linear light + one mapper",
    tracer=al.Tracer(
        galaxies=[
            lens,
            al.Galaxy(redshift=1.0, pixelization=pixelization(shape=(3, 3))),
        ]
    ),
    rtol_reconstruction=1.0e-11,
    rtol_log_evidence=1.0e-12,
)

"""
__Linear Light + Two Mappers__

Two source planes at different redshifts, each with its own pixelization, so
the MAPPER-MAPPER off-diagonal blocks are exercised too.

Fitting ONE dataset with TWO free pixelized components is degenerate by
construction — `cond(F + H)` sits at the float64 floor (~7e16) for every
geometry tried (mask radius 0.6-0.8", Einstein radius 0.3-0.8, second source
plane z = 2-5), so `F + H` is numerically singular no matter how the model is
posed. `curvature_matrix` and `data_vector` are unaffected and still agree at
round-off (4.2e-16 / 4.2e-16, identical to the one-mapper block), but the solve
that consumes them amplifies that round-off by an arbitrary amount: measured
5.2e-7 for `reconstruction` and 6.0e-8 for `log_evidence` here, and anywhere in
7e-8 - 1.2e-5 across the geometries scanned.

The two tolerances below are therefore deliberately loose SANITY bounds, not
precision gates — pinning them near the measured value would buy a flaky test,
not extra coverage. What actually gates #500's arithmetic in this block is the
fixed 1e-12 assertion on `curvature_matrix`, where the mapper-mapper
off-diagonals live: dropping those blocks (the pre-#500 behaviour) is an O(1)
relative error, not a 1e-5 one.
"""
assert_parity(
    name="linear light + two mappers",
    tracer=al.Tracer(
        galaxies=[
            lens,
            al.Galaxy(redshift=1.0, pixelization=pixelization(shape=(3, 3))),
            al.Galaxy(redshift=2.0, pixelization=pixelization(shape=(4, 4))),
        ]
    ),
    rtol_reconstruction=1.0e-3,
    rtol_log_evidence=1.0e-5,
)

print("fit_interferometer_sparse_operator: all assertions passed")
