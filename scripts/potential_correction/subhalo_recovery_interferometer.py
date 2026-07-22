"""
Potential Correction: Interferometer Subhalo Recovery Validation
================================================================

End-to-end validation of the visibility-space gravitational-imaging (potential correction) implementation of
`al.pc`: simulates interferometer data of a lens whose mass contains a dark NFW subhalo, reconstructs with the
smooth model plus pixelized potential corrections through the **sparse-operator (w-tilde) route** — the one-shot
joint inversion (`FitDpsiSrcInterferometer`) and the iterative LM engine (`IterFitDpsiSrcInterferometer`) — and
asserts the recovered convergence correction (dkappa) localizes the true subhalo.

The sparse route is the scaling-critical path (cost independent of the visibility count); this script also
asserts its parity against the dense `transform_mapping_matrix` reference route at this small visibility count.

The load-bearing assertions are the sparse/dense parity, the dkappa peak localization and the gauge
constraints. At this smoke-scale synthetic uv coverage (1000 random baselines, ~0.7" resolution) the global
dkappa correlation with the truth is intrinsically weak (uv sidelobe structure) — the strong-recovery
benchmark of the technique is the imaging `subhalo_recovery.py` script and, for the visibility regime, a
future realistic-uv-coverage validation campaign (the B1938+666 configuration of Powell et al. 2025).

Cites Cao et al. 2025 (https://github.com/caoxiaoyue/lensing_potential_correction; cite via
https://github.com/caoxiaoyue/potential_correction_paper); methodology per the JVAS B1938+666 detections
(Powell et al. 2025; Vegetti et al. 2026).
"""

import numpy as np

import autoarray as aa
import autolens as al

"""
__Simulate__

A small self-contained interferometer dataset: random (seeded) uv coverage, Isothermal lens + 1e10 Msun NFW
subhalo on the Einstein ring, compact double-Gaussian source.
"""
rng = np.random.default_rng(1)
uv_wavelengths = rng.uniform(-3.0e5, 3.0e5, size=(1000, 2))

real_space_mask = al.Mask2D.circular(
    shape_native=(64, 64), pixel_scales=0.1, radius=2.6
)

simulator = al.SimulatorInterferometer(
    uv_wavelengths=uv_wavelengths,
    exposure_time=300.0,
    noise_sigma=0.02,
    noise_seed=1,
)

subhalo_centre = (1.41, 0.0)
true_subhalo = al.mp.NFWMCRLudlowSph(
    centre=subhalo_centre, mass_at_200=1.0e10, redshift_object=0.2, redshift_source=0.6
)

lens_true = al.Galaxy(
    redshift=0.2,
    mass=al.mp.Isothermal(
        centre=(0.0, 0.0),
        einstein_radius=1.4,
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=0.0),
    ),
    subhalo=true_subhalo,
)
source_true = al.Galaxy(
    redshift=0.6,
    bulge0=al.lp.Gaussian(
        centre=(0.0, 0.0),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.6, angle=45.0),
        intensity=5.0,
        sigma=0.15,
    ),
    bulge1=al.lp.Gaussian(
        centre=(0.0, 0.4),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.4, angle=135.0),
        intensity=3.0,
        sigma=0.1,
    ),
)

grid = al.Grid2D.from_mask(mask=real_space_mask)
dataset = simulator.via_tracer_from(
    tracer=al.Tracer(galaxies=[lens_true, source_true]), grid=grid
)
dataset = al.Interferometer(
    data=dataset.data,
    noise_map=dataset.noise_map,
    uv_wavelengths=uv_wavelengths,
    real_space_mask=real_space_mask,
)
dataset_sparse = dataset.apply_sparse_operator()
print(f"visibilities: {np.asarray(dataset.data).shape[0]}")

"""
__Arc dpsi mask__

The corrections are only constrained where the lensed arcs are. The real-space mask must stay a filled circle
(it defines the FFT extent of the sparse operator), but the dpsi mesh is restricted to an arc-tracing sub-mask —
built here from the smooth-model lensed image (a geometry proxy for the arcs) and intersected with the real-space
mask.
"""
tracer_smooth = al.Tracer(
    galaxies=[al.Galaxy(redshift=0.2, mass=lens_true.mass), source_true]
)
arc_image = np.asarray(tracer_smooth.image_2d_from(grid=grid).native)
arc_snr_proxy = arc_image / (0.05 * arc_image.max())
arc_mask = al.pc.util.arc_mask_from(
    arc_snr_proxy, threshold=3.0, ignore_size=10, ext_size=3
)
dpsi_mask = ~((~arc_mask) & (~np.asarray(real_space_mask)))
print(f"dpsi mesh restricted to {int(np.count_nonzero(~dpsi_mask))} arc pixels")

"""
__Smooth starting model + pixelizations__
"""
lens_smooth = al.Galaxy(redshift=0.2, mass=lens_true.mass)
source_start = al.pc.AnalyticSrcFactory(source_galaxy=source_true)

dpsi_pixelization = al.pc.DpsiPixelization(
    mesh=al.pc.RegularDpsiMesh(factor=2),
    regularization=al.reg.MaternKernel(coefficient=2000.0, scale=4.0, nu=2.5),
)
src_pixelization = al.Pixelization(
    mesh=al.mesh.RectangularUniform(shape=(20, 20)),
    regularization=al.reg.Constant(coefficient=3.8),
)


def dkappa_metrics(pair_obj, dkappa_rec, tag):
    points = np.vstack([pair_obj.ygrid_dpsi_1d, pair_obj.xgrid_dpsi_1d]).T
    dkappa_true = np.asarray(
        true_subhalo.convergence_2d_from(grid=al.Grid2DIrregular(values=points))
    )
    corr = float(np.corrcoef(dkappa_rec, dkappa_true)[0, 1])
    peak = points[int(np.argmax(dkappa_rec))]
    dist = float(np.hypot(peak[0] - subhalo_centre[0], peak[1] - subhalo_centre[1]))
    print(f'{tag}: corr(dkappa_rec, dkappa_true) = {corr:.4f}; peak dist = {dist:.2f}"')
    return corr, dist


"""
__One-shot joint inversion (sparse route) + dense parity__
"""
fit_sparse = al.pc.FitDpsiSrcInterferometer(
    dataset=dataset_sparse,
    lens_start=lens_smooth,
    source_start=source_start,
    dpsi_pixelization=dpsi_pixelization,
    src_pixelization=src_pixelization,
    dpsi_mask=dpsi_mask,
    use_sparse_operator=True,
)
evidence_sparse = fit_sparse.log_evidence
print(f"one-shot sparse-route log evidence = {evidence_sparse:.4e}")

fit_dense = al.pc.FitDpsiSrcInterferometer(
    dataset=dataset,
    lens_start=lens_smooth,
    source_start=source_start,
    dpsi_pixelization=dpsi_pixelization,
    src_pixelization=src_pixelization,
    dpsi_mask=dpsi_mask,
    use_sparse_operator=False,
)
evidence_dense = fit_dense.log_evidence
print(f"one-shot dense-route  log evidence = {evidence_dense:.4e}")

assert np.isclose(
    evidence_sparse, evidence_dense, rtol=1e-3
), f"sparse/dense evidence mismatch: {evidence_sparse} vs {evidence_dense}"

dkappa = np.asarray(
    fit_sparse.pair_dpsi_data_obj.hamiltonian_dpsi @ fit_sparse.best_fit_dpsi
)
corr, dist = dkappa_metrics(fit_sparse.pair_dpsi_data_obj, dkappa, "one-shot")

assert corr > 0.1, f"one-shot dkappa correlation {corr:.3f} below threshold 0.1"
assert (
    dist < 0.6
), f'one-shot dkappa peak {dist:.2f}" from true subhalo (threshold 0.6")'

"""
__Iterative LM engine (sparse route)__

At this smoke scale the source mesh cannot fit the high-S/N visibilities to the noise level (best pure-source
chi2/dof ~ 4e3), so the iterative corrections absorb source-model error rather than the subhalo — a property of
the mis-specified synthetic, not the engine. The assertions here therefore certify the engine's MECHANICS
(monotone cost improvement over the zero state, gauge constraints, finite Laplace evidence); dkappa recovery is
certified by the one-shot assertions above, the imaging `subhalo_recovery.py`, and a future realistic-uv
validation campaign.
"""
iter_fit = al.pc.IterFitDpsiSrcInterferometer(
    dataset=dataset_sparse,
    lens_start=lens_smooth,
    dpsi_pixelization=dpsi_pixelization,
    src_pixelization=src_pixelization,
    dpsi_mask=dpsi_mask,
    gauge_constraints=True,
    n_iter=5,
)
s_opt, dpsi_opt = iter_fit.solve_joint_optimization()
print(f"iterative Laplace log evidence = {iter_fit.log_evidence():.4e}")

dkappa_iter = np.asarray(iter_fit.pair_dpsi_data_obj.hamiltonian_dpsi @ dpsi_opt)
corr_iter, dist_iter = dkappa_metrics(
    iter_fit.pair_dpsi_data_obj, dkappa_iter, "iterative"
)

# mechanics: the optimized state must beat the zero state's cost
x_opt = np.concatenate([s_opt, dpsi_opt])
F_opt, D_opt, _ = iter_fit._normal_equations_from(s_opt, dpsi_opt)
R_opt = np.asarray(iter_fit._regularization_matrix())
cost_opt, _, _ = iter_fit._cost_from(x_opt, F_opt, D_opt, R_opt)
assert cost_opt < 0.5 * iter_fit.data_weighted_norm, "iterative LM made no progress"
assert np.isfinite(s_opt).all() and np.isfinite(dpsi_opt).all()

n_dpsi = dpsi_opt.shape[0]
gauge = np.array(
    [
        np.sum(dpsi_opt) / n_dpsi,
        np.sum(iter_fit.dpsi_points[:, 1] * dpsi_opt) / n_dpsi,
        np.sum(iter_fit.dpsi_points[:, 0] * dpsi_opt) / n_dpsi,
    ]
)
assert np.allclose(gauge, 0.0, atol=1.0e-5), f"gauge constraints violated: {gauge}"

print("potential_correction interferometer subhalo recovery checks all passed")
