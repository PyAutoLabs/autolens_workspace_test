"""
Potential Correction: Subhalo Recovery Validation
=================================================

End-to-end validation of the gravitational-imaging (potential correction) implementation of `al.pc`: simulates
imaging of a lens whose mass model contains a dark NFW subhalo, fits it with the smooth (subhalo-free) model plus
pixelized potential corrections — one-shot joint inversion and the iterative Levenberg-Marquardt engine — and
asserts the recovered convergence correction (dkappa) localizes the true subhalo.

This is the workspace-level parity anchor of the port of Cao et al. 2025's `potential_correction` package
(https://github.com/caoxiaoyue/lensing_potential_correction; cite via
https://github.com/caoxiaoyue/potential_correction_paper), mirroring its demo reconstruction. On the reference
200x200 demo dataset the merged implementation recovers corr(dkappa_rec, dkappa_true) = 0.77 with the peak 0.21"
from the true subhalo (one-shot); the reduced problem here asserts conservative thresholds.

__Env__

Test-harness configuration (PyAutoHands docs/env_profile_redesign.md §10).
Simulates a 120x120 dataset inline with resolution-calibrated dkappa
thresholds; the 15x15 cap breaks the mesh geometry.

ENV: full_datasets
"""

import numpy as np

import autolens as al

"""
__Simulate__

Isothermal lens + 1e10 Msun NFW subhalo on the Einstein ring; compact double-Gaussian source (sharp gradients are
what gravitational imaging leverages).
"""
grid = al.Grid2D.uniform(shape_native=(120, 120), pixel_scales=0.05, over_sample_size=2)
psf = al.Convolver.from_gaussian(shape_native=(11, 11), sigma=0.05, pixel_scales=0.05)

simulator = al.SimulatorImaging(
    exposure_time=840.0,
    psf=psf,
    background_sky_level=0.1,
    add_poisson_noise_to_data=True,
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

dataset = simulator.via_tracer_from(
    tracer=al.Tracer(galaxies=[lens_true, source_true]), grid=grid
)

mask_array = al.pc.util.arc_mask_from(
    np.asarray(dataset.signal_to_noise_map.native),
    threshold=3.0,
    ignore_size=25,
    ext_size=5,
)
masked_imaging = dataset.apply_mask(
    mask=al.Mask2D(mask=mask_array, pixel_scales=dataset.pixel_scales)
)
print(f"unmasked pixels: {int(np.count_nonzero(~mask_array))}")

"""
__Smooth starting model + pixelizations__
"""
lens_smooth = al.Galaxy(redshift=0.2, mass=lens_true.mass)
source_start = al.pc.AnalyticSrcFactory(source_galaxy=source_true)

dpsi_pixelization = al.pc.DpsiPixelization(
    mesh=al.pc.RegularDpsiMesh(factor=2),
    regularization=al.reg.MaternKernel(coefficient=2000.0, scale=4.0, nu=2.5),
)

grid_slim = masked_imaging.grid.slim
source_shape = (
    int(float(grid_slim[:, 0].max() - grid_slim[:, 0].min()) / 0.05 / 2.0),
    int(float(grid_slim[:, 1].max() - grid_slim[:, 1].min()) / 0.05 / 2.0),
)
src_pixelization = al.Pixelization(
    mesh=al.mesh.KNearestNeighbor(pixels=int(np.prod(source_shape))),
    regularization=al.reg.Constant(coefficient=3.8),
)
src_image_mesh = al.image_mesh.Overlay(shape=source_shape)


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
__One-shot joint inversion__

The recovered dkappa must correlate with the true subhalo convergence and peak near its position.
"""
fit = al.pc.FitDpsiSrcImaging(
    masked_imaging=masked_imaging,
    lens_start=lens_smooth,
    source_start=source_start,
    dpsi_pixelization=dpsi_pixelization,
    src_pixelization=src_pixelization,
    src_image_mesh=src_image_mesh,
)
print(f"one-shot log evidence = {fit.log_evidence:.4e}")

dkappa = np.asarray(fit.pair_dpsi_data_obj.hamiltonian_dpsi @ fit.best_fit_dpsi)
corr, dist = dkappa_metrics(fit.pair_dpsi_data_obj, dkappa, "one-shot")

assert corr > 0.5, f"one-shot dkappa correlation {corr:.3f} below threshold 0.5"
assert (
    dist < 0.5
), f'one-shot dkappa peak {dist:.2f}" from true subhalo (threshold 0.5")'

"""
__Iterative LM engine__

The iterative reconstruction (re-ray-tracing through the corrected lens each accepted step, gauge-constrained)
must also localize the subhalo. It is warm-started from the one-shot joint solution `[s | dpsi]` — the certified
recipe (PyAutoLens#630, mirroring the interferometer uv campaign #627): under the Marquardt-scale LM damping
`mu * diag(H)`, cold-starting from zeros no longer reliably lands in the subhalo basin, but refining inside the
one-shot basin does. The LM then confirms the one-shot correction is a genuine cost minimum rather than drifting
off it. `damping="marquardt"` is pinned explicitly: PyAutoLens#676 changed the imaging default to `"identity"`,
whose near Gauss-Newton trial steps are rejected from the warm-started optimum — a rejection storm (each trial a
full Jacobian rebuild) that pushed this script past the 300s smoke cap without changing the result.

Each accepted LM step re-imposes the gauge constraints <dpsi,1> = <dpsi,x> = <dpsi,y> = 0 on the updated state,
but from the (un-gauged) one-shot solution the gauge-fixing move slides along the model-degenerate constant /
deflection-drift modes and is not cost-decreasing, so it is rejected and the solve would stall off the constraint
surface. Passing `gauge_project_x0=True` has the engine project those modes out of x0 first; they are the null
space of the Hamiltonian dpsi -> dkappa map, so the recovered convergence correction (and every assertion below)
is unchanged while the gauge-constrained solve stays on the constraint surface.
"""
iter_fit = al.pc.IterFitDpsiSrcImaging(
    masked_imaging=masked_imaging,
    lens_start=lens_smooth,
    dpsi_pixelization=dpsi_pixelization,
    src_pixelization=src_pixelization,
    src_image_mesh=src_image_mesh,
    gauge_constraints=True,
    damping="marquardt",
    n_iter=3,
)
x0 = np.concatenate([np.asarray(fit.best_fit_source), np.asarray(fit.best_fit_dpsi)])
s_opt, dpsi_opt = iter_fit.solve_joint_optimization(x0=x0, gauge_project_x0=True)
print(f"iterative Laplace log evidence = {iter_fit.log_evidence():.4e}")

dkappa_iter = np.asarray(iter_fit.pair_dpsi_data_obj.hamiltonian_dpsi @ dpsi_opt)
corr_iter, dist_iter = dkappa_metrics(
    iter_fit.pair_dpsi_data_obj, dkappa_iter, "iterative"
)

n_dpsi = dpsi_opt.shape[0]
gauge = np.array(
    [
        np.sum(dpsi_opt) / n_dpsi,
        np.sum(iter_fit.dpsi_points[:, 1] * dpsi_opt) / n_dpsi,
        np.sum(iter_fit.dpsi_points[:, 0] * dpsi_opt) / n_dpsi,
    ]
)
assert np.allclose(gauge, 0.0, atol=1.0e-5), f"gauge constraints violated: {gauge}"
assert (
    corr_iter > 0.3
), f"iterative dkappa correlation {corr_iter:.3f} below threshold 0.3"
assert (
    dist_iter < 0.7
), f'iterative dkappa peak {dist_iter:.2f}" from true subhalo (threshold 0.7")'

print("potential_correction subhalo recovery checks all passed")
