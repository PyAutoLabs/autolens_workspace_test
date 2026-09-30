# PointSolver error audit — cluster arc phase 1a

Audit date: 2026-09-30. Task: [#328](https://github.com/PyAutoLabs/autolens_workspace_test/issues/328).
This is numerical audit evidence, not a declaration that the solver is trusted.
The parent phase still needs hardening and review before later arc phases start.

## Findings and decisions

1. **Keep local point magnification as the filter quantity.** The old
   scale-sized finite-difference Hessian changes magnification estimates, not
   the coordinates of a surviving triangle centroid. In the near-caustic
   fixture at scale 0.2 and precision 0.0001, a negative-parity image has
   mu=-151.19 with the old scale-sized step versus -288.23 with current local
   differentiation. The step=0.01 replay gives -287.55. Threshold 200 therefore
   changes image membership. At threshold 0.1, the old and new estimates keep
   the same candidates in these fixtures. This does **not** demonstrate that
   the Hessian change caused the user's historical position-error change.
   A finite-difference step tied to a triangle edge is not an integration over
   a finite source/image area; finite-area magnification belongs to the later
   area-magnification phase.
2. **Do not lower the threshold blindly.** The singular lens centre produces a
   spurious candidate with a large lens-equation residual. Threshold 0.1
   rejects it, while 1e-8 retains it here. The old step=0.2 estimate assigns it
   |mu|~0.0205; the current local value at fine precision is ~0.0000675.
   At threshold 0.01 this too changes membership. Raw candidate errors are
   therefore recorded separately from errors of the filtered physical images.
3. **Default padding is inconsistent with the documented xp override.** A
   solver constructed with `use_jax=False` and called with `xp=jnp` still uses
   `remove_infinities = not self.use_jax`. Its eager output is stripped; the
   same default call under jit raises `NonConcreteBooleanIndexError`. Explicit
   `remove_infinities=False` succeeds. The JSON records the failed default
   call and the explicitly padded measurement separately. This needs a
   focused library regression/fix; no library has been changed by this audit.
4. **Symmetric images can be duplicated.** At initial scale 0.05 the analytic
   quad's NumPy/eager-JAX paths return six filtered rows, representing four
   image groups. JIT yields four in this fixture. The extra centroids straddle
   two true images on a triangle boundary. Grouping within twice the requested
   precision is a diagnostic only; raw positions/counts remain in the JSON.
   A production deduplication rule must consider real close image pairs and
   cannot simply adopt this audit's grouping radius.
5. **Capacity overflow remains silent.** Twenty-four distinct nested triangles
   enclosing the origin return only 15 or 20 indices at those caps, in eager
   and jit execution, without an exception. Both astrophysical fixtures stay
   below the old cap: maximum 8 for the quad, 13 for near-caustic. Thus capacity
   does not explain this audit's physical-fixture discrepancy, but cluster
   safety still needs an overflow signal and a deliberate capacity policy.
6. **The float32 placeholders did not downcast valid float64 data.** Both
   `ArrayTriangles.triangles` and `for_indexes` preserve float64 vertices in
   eager/jit tests with x64 enabled. Float32 inputs remain float32. The
   placeholder literals alone do not establish a precision defect. This does
   not test every dtype path or historical x64-disabled environment.

## Corrected history

| Boundary | What the source actually changes | Evidence limitation |
|---|---|---|
| PyAutoArray e0e2f28e (2025-11-03) and 314e2d09; PyAutoLens 0ea7c6000/c36f8a6ec | x64 side-effect removal and NumPy/JAX triangle refactoring | Inspected source; no full old-stack numerical attribution |
| PyAutoLens dd82ce386 (2025-11-18) | Adds default infinity stripping after solving/filtering | Shape/padding change; stripping does not move finite rows |
| PyAutoLens 14826f6c7 (2026-03-02), 3db51dd38 (03-04) | Drops the explicit scale-sized Hessian buffer during composition/API migration | Historical Hessian method replay verified numerically on current deflections |
| PyAutoGalaxy ee92bebe (2026-03-02) | Adds exact JAX jacfwd Hessian | Earlier than the April date in the intake hypothesis |
| PyAutoGalaxy eacdcd77 (2026-04-18) | Adds NumPy Richardson extrapolation | Current NumPy also includes adaptive refinement from #591 |
| PyAutoLens 5c42d8133 (2024-12-16) | Removes configurable max_containing_size from solver factories | The actual removal predates the proposed window |
| PyAutoLens fca58c468 (2026-04-21) | Refactors xp and removes stale max_containing_size documentation | Does not remove that already-absent argument |
| PyAutoLens d24339c37 (2026-05-24) | Adds constructor-backend-dependent padding defaults | Current explicit-xp override inconsistency reproduced |

Full SHAs and parent SHAs are in `history_evidence.json`. The archived
pre-March import attempt used Lens `2aba5a66d70d13baf9abd08665a6249bdfb1c6e7`,
Array `9bfde26c794734eabac2669d1d09b4b983d2df7a`, Galaxy
`8d7747d8ee5486fcbe9da0f137dc2f94640ea332`, and Fit
`ddad8283b2381b4967ea703c5816c2770ad49626` (upstreams selected by commit date).
It failed immediately with `ModuleNotFoundError: No module named 'autoconf'`.
These date-selected sources are not a recovered historical lockfile. The shared
Python installation was left intact. No old full-stack numerical run is claimed.

`history_audit.py` extracts the original undecorated `hessian_from` function
from that Galaxy revision and compares it with the explicit central-difference
replay at buffers `scale` and `0.01`, using current deflections. This isolates
the Hessian-step mechanism; it cannot rule out other historical solver changes.

## Fixtures, references and interpretation

- **Analytic quad:** circular isothermal theta_E=1.6 plus external shear
  gamma_1=0.1, gamma_2=0; source at (0,0). Reference images in (y,x) order are
  (+/-1.6/1.1,0) and (0,+/-1.6/0.9). Reference lens-equation residual <1e-8.
- **Near-caustic:** q=0.9, angle=45 degrees, theta_E=1.6; source frozen at
  (0.07777695929510982,0.07777695971376672). Provenance:
  `autolens_profiling` commit `3ad68afad4dcc5f4d0689fac57620bef86a0e22d`,
  `dataset/point_source/near_caustic/truth.json` and
  `scripts/misc/simulators/point_source.py` (95% of the diagonal caustic).
  Reference roots use 128 angular initial guesses at radius 1.6, solved at
  tolerances 1e-9 and 1e-11; both yield the same four roots within 1e-8,
  with source-plane residual <1e-9. This is independent of the triangle solver,
  but not a global proof of root completeness for arbitrary lenses.
- Both use lens z=0.5, source z=1.0, requested limits [-4.97,5.03] on each axis,
  scales 0.2/0.05, precisions 0.001/0.0001; current x64 CPU execution.
  Libraries' exact SHAs, package versions and device are in `current_evidence.json`.
- Matching uses minimum-distance assignment, never array row order. Unequal
  cardinality gives a null positional error rather than hiding extra images.
  `diagnostic_unique_error` reports the separate grouping check.
- Signed mu, raw positions, shapes, dtypes, residuals, threshold survival indices
  and counts are preserved. Times include compilation and are not benchmarks.
- The isothermal singularity can emit an arctanh divide-by-zero warning during
  root exploration/refinement. Raw central candidates and their residuals remain
  visible; the physical-image reference assertions still have to pass.

## Reproduce

Activate a task environment pointing to the JSON's library revisions. From the
workspace root, enable x64 and full datasets, then run:

```bash
export JAX_ENABLE_X64=True
unset PYAUTO_SMALL_DATASETS
python scripts/point_source/solver/error_audit.py --output scripts/point_source/solver/current_evidence.json
python scripts/point_source/solver/mechanism_audit.py --output scripts/point_source/solver/mechanism_evidence.json
python scripts/point_source/solver/history_audit.py --output scripts/point_source/solver/history_evidence.json
```

The main audit supports `--resume` with identical environment/library pins,
`--scales`, `--precisions`, and `--backends numpy jax jit`. Each completed row is
checkpointed. Audit findings are data, not assertions that buggy behavior is
correct. Reference convergence, historical replay agreement, actual filter vs
predicted filter agreement and dtype/capacity witnesses are executable checks.
Historical-method replay requires full local git history. These scripts are
opt-in and are not added to the routine smoke allowlist. This integration-test
repository has no tracked notebook surface to regenerate.

## Remaining phase-1 scope

First fix/test padding selection from the effective call-time backend. Then
settle duplicate-image semantics and add analytic-quad positional/count
regressions covering boundary alignment. Add a loud containment-overflow signal
with a deliberate policy for cluster capacity. The float32 literals are not
justification for a numerical fix on their own. Full historical-stack recovery
remains necessary if exact attribution of the original error report is required.
No later arc phase is unlocked by this audit alone, and no follow-up issue is
bulk-filed here.

## Numerical summary

Counts are actual filtered rows (NumPy / eager JAX / JIT). JIT uses explicit
padding after recording the default-call exception. The error column uses the
four diagnostic image groups when raw cardinality differs.

| Fixture | Initial scale | Precision | Counts | Max grouped error (arcsec) |
|---|---:|---:|---|---:|
| analytic_quad | 0.2 | 0.001 | 4 / 4 / 4 | 0.00078233 |
| analytic_quad | 0.2 | 0.0001 | 4 / 4 / 4 | 0.00007176 |
| analytic_quad | 0.05 | 0.001 | 6 / 6 / 4 | 0.00078233 |
| analytic_quad | 0.05 | 0.0001 | 6 / 6 / 4 | 0.00007176 |
| near_caustic | 0.2 | 0.001 | 4 / 4 / 4 | 0.00076414 |
| near_caustic | 0.2 | 0.0001 | 4 / 4 / 4 | 0.00006342 |
| near_caustic | 0.05 | 0.001 | 4 / 4 / 4 | 0.00076414 |
| near_caustic | 0.05 | 0.0001 | 4 / 4 / 4 | 0.00006342 |

All 24 rows resolve four diagnostic groups. All eight default JIT calls on a
NumPy-constructed solver record the padding exception. Decreasing requested
precision from 0.001 to 0.0001 reduces grouped errors from about 0.00076–0.00078
to 0.000063–0.000072 arcsec for these fixtures. Grouping is not a solver fix.

## Validation

- Main audit: 24 measured rows; reference and filter-consistency checks passed.
- Historical Hessian replay: all 16 buffer comparisons agree within 1e-12.
- Capacity/dtype witnesses passed; physical per-step counts recorded.
- Existing `scripts/point_source/jax_likelihood/image_plane.py`: exit 0.
- Repository `.github/scripts/run_smoke.py`: **32 passed, 0 failed**, CPU local environment.
- Formatting checked with Black; JSON parses and the table was checked against all 24 rows.
- GitHub Python 3.12/3.13 PR CI has not run: the branch is not yet published.

Raw logs live under the task worktree `../scratch/`; smoke machine report is
`test-results/autolens_workspace_test__scripts__script.json`. These are ignored
runtime artifacts. Numerical witnesses and version pins are retained beside
this report.
