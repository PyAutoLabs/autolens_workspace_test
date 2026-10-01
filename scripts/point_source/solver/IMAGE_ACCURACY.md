# PointSolver image-position accuracy — cluster arc phase 1d

Task: [autolens_workspace_test#333](https://github.com/PyAutoLabs/autolens_workspace_test/issues/333).
This is a bounded numerical research artifact. No production library is changed.


## Decision

**NO-GO for a production acceptance or image-identity rule based on these
bounded refinements and local conditioning screens.** The complete 32-row
matrix preserves the phase-1c counterexamples and adds a resolved-pair false
accept. No production policy is promoted. Requiring the polisher's convergence
flag rejects the false accepts but then misses reference images in all eight
closest-cusp rows. A diagnostic should expose those outcomes as unresolved.

## Question and method

Phase 1c established uncapped NumPy near-cusp position errors independently of
JAX truncation. This cell measures whether three additional triangle refinement
levels or conditioning-aware polishing diagnostics settle that problem.

The matrix contains 32 rows: the centred SIS plus shear quad and the three
existing SIE cusp offsets (inward fractions 1e-3, 1e-5, 1e-7), at requested
precisions 1e-3 and 1e-4 arcsec, each with zero through three extra levels.
Initial scale is 0.2 arcsec. Each extra level halves the target precision.
All production solves and step traversals use uncapped NumPy on CPU in fp64.
The angular-root oracle from phase 1c is reused, checked at two angular grid
resolutions, and cross-checked against the analytic quad. It is specific to
these fixtures, not a general cluster-lens completeness proof.

The cell records per-step retained triangle counts, edge sizes, root membership
in straight image-plane triangles and nearest-centroid distance to every root.
Terminal geometry is interpreted only after applying the production
magnification filter and verifying one-to-one centroid correspondence with a
separate production solve. The filter removes a central candidate in these
fixtures. Membership is an exact floating-point straight-triangle test; a
reported absence can reflect a boundary, nonlinear mapping or numerical error.
It is not by itself proof of irrecoverable image loss. Later neighbourhood
expansion can recover membership.

For every candidate the evidence stores its position, nearest-root label/error,
source residual, parity and singular values of the lens-equation Jacobian.
`LensCalc.jacobian_from` returns (x,y) components, so both axes are swapped
before applying the matrix to (y,x) vectors. An independent central finite
difference of the NumPy residual checks the matrix at the reference roots.
The norm of the linearized correction `A^-1 r` is compared with actual error;
it is a diagnostic, not a nonlinear error bound or uniqueness certificate.

Polishing uses SciPy `root` with tolerance 1e-11 and `maxfev=100`. Solver status,
evaluation count, displacement and finite-output status are saved. No roots
are grouped. The residual-only screen is `|r| < 1e-10`. The conditioning screen
additionally requires a nonsingular diagnostic Jacobian (smallest singular
value above 1e-14) and `|A^-1 r| <= target_precision`. These screens are
explicitly checked against oracle errors; neither requires SciPy's success
flag, which is recorded independently. An accepted candidate is false when its
nearest-root error exceeds **one** target precision. Coverage requires a
candidate within that tolerance of **each** independent root, not four rows.
This tolerance differs from phase 1c's two-precision grouping criterion.
The summary also checks injective assignment to four distinct candidates and
labels multiplicity unresolved when the minimum root separation is no greater
than twice target precision. Even injective coverage within that tolerance is
not a root-identity proof.

## Reproduce and inspect

From the workspace root with the pinned libraries and CPU JAX x64 environment:

```bash
python scripts/point_source/solver/image_accuracy.py
python scripts/point_source/solver/image_accuracy.py --summarize
```

The JSON pins this script and both imported helper files by SHA256, records
library SHAs/dirty state and package versions, and checks identical provenance
before and after the full matrix. The summarizer checks the exact 32-row key
set, reference quad, step counts and diagnostic shapes/statuses; it recomputes
the summary and never rewrites the evidence. No phase-1c evidence is rewritten.
The traversal stops further work if the 200,000-triangle budget is exceeded,
recording the row as resource-limited; such rows cannot count as passes.
The full research sweep remains opt-in, outside the smoke allowlist.

## Measurements

All **32/32 rows completed**, with no resource-limited rows and unchanged
before/after script/helper/library/environment provenance. Terminal geometry
matched the filtered production output in every row. Measured row runtimes
sum to 202 seconds (diagnostic work included; not a controlled speed benchmark).
The largest reference-Jacobian finite-difference discrepancy is 2.69e-11.

At the fine base precision of 1e-4 arcsec:

| Cusp fraction | Extra levels | Target precision | Candidates | Worst raw image error | Roots covered by raw candidates |
|---|---:|---:|---:|---:|---:|
| 0.001 | 0 | 0.0001 | 4 | 8.09e-05 | 4/4 |
| 0.001 | 3 | 1.25e-05 | 4 | 4.02e-06 | 4/4 |
| 1e-05 | 0 | 0.0001 | 86 | 0.00352 | 3/4 |
| 1e-05 | 3 | 1.25e-05 | 12 | 4.18e-05 | 3/4 |
| 1e-07 | 0 | 0.0001 | 82 | 0.00487 | 3/4 |
| 1e-07 | 3 | 1.25e-05 | 178 | 0.000816 | 3/4 |

Across the full matrix, **10/32 raw rows fail per-root coverage** at their own
target precision. The closest-cusp fine row grows from 82 to 178 candidates
while its worst error falls from 4.87e-3 to 8.16e-4 arcsec. The final target is
1.25e-5: the worst candidate is still about 65 times too far away. Counts and
errors therefore cannot be treated as a completeness or precision certificate.
This is uncapped NumPy evidence; none of these rows truncates at JAX's cap.

The residual-only polished screen falsely accepts 447 candidates across rows;
adding the linearized-correction screen reduces this to three candidates in
three closest-cusp rows. The latter is useful rejection evidence, but it is
not sufficient for production. One counterexample has separated reference roots:

- cusp fraction 1e-7, base precision 1e-3, two extra levels;
- target precision 2.50e-4; nearest reference separation 7.06e-4 arcsec;
- polished position `(1.1916800011902005, 1.1901528610703465)`;
- source residual 9.94e-12 and smallest singular value 4.13e-8;
- predicted correction 2.402e-4, but actual nearest-root error **3.736e-4**.

All three false accepts have `success=False`, status 2 (evaluation budget),
with 102 reported evaluations for `maxfev=100` (MINPACK's requested budget,
not a strict Python-call cutoff). Adding `success` to the screen rejects all
three. However, that stricter screen covers only one or two of four reference
images in **every closest-cusp row**, including the fine extended rows. A
failed convergence flag must not be silently ignored to recover completeness.
These stricter-screen counts are derived from the saved `success`,
`condition_accept`, positions and per-root reference positions; no new solve.

Near the cusp, a linearized correction can underestimate nonlinear positional
error. This experiment rejects the tested acceptance rule, not all possible
conditioning-aware methods. The independently recovered roots and numerical
convergence checks remain fixture-specific, not interval certificates.

Step histories show intermittent reference membership in the retained straight
triangles. For the resolved counterexample above, the three cusp roots are
absent at step 0, all present at step 1, one absent at steps 6–7, all present at
step 8, and one absent at step 9. Boundary absences also occur in the analytic
quad while position errors remain small. Thus an early membership failure
alone does not establish irreversible loss or isolate a unique implementation
fault. The evidence identifies persistent inaccurate candidates across terminal
refinements; it does not prove whether curvature, containment arithmetic or
another mechanism dominates each one.

## Next bounded work and limits

Keep the image-accuracy/identity and containment-overflow contracts separate.
The next production task should make containment overflow observable under
eager/JIT/vmap with an explicit invalid-result policy, after the PyAutoArray
claim clears and its API plan is approved. That guard will not repair these
uncapped positional failures. No follow-up issue is queued here.

For accuracy, a future bounded experiment should test a safeguarded local
refinement or nonlinear error bound against these exact failed cases, requiring
convergence, per-root coverage, unresolved-multiplicity handling and eventual
registered-tracer/JIT/vmap compatibility before promoting an identity rule.
Merely increasing the number of refinement levels or tightening a fixed
source-residual cutoff is not established as safe by this evidence.

Scope remains two fixtures, CPU fp64, forward solutions. No GPU, mixed
precision, general cluster completeness or implicit-gradient claim is made.
Parent arc phase 1 remains incomplete; phase 2 remains gated. Cortex phase 11
remains dropped and no science project is born by this numerical dev task.

## Validation

- Final-revision matrix: 32/32 complete, geometry correspondence 32/32,
  before/after script/helper/library/environment provenance PASS.
- Saved summary/read-only validation PASS; four corrupt-evidence controls
  (missing row, duplicate row, wrong helper hash, NaN metric) are rejected.
  Raw and polished distances/coverage independently recompute from saved
  positions; explicit empty-candidate metrics/polishing controls pass.
- Existing padding/backend matrix: 12/12 PASS.
- Existing image-plane/JIT regression: PASS, including registered fit round trips.
- Full workspace smoke: 32/32 PASS, zero failures.
- Black, compilation, strict JSON and diff-whitespace checks PASS; in-session
  review complete (not an independent review-faculty judgment).
- Logs in the task bundle's `scratch/`: `image-accuracy-final.log`,
  `evidence-validation.log`, `summary.log`, `padding.log`, `image-plane.log`,
  `smoke.log`, `review.md`, `vitals.log`, `readiness.json`.
  The workspace smoke machine report is
  `test-results/autolens_workspace_test__scripts__script.json`.

The first exploratory run was stopped to correct terminal geometry matching;
a subsequent complete pass prompted explicit multiplicity/injective-coverage
fields. Only the final full rerun with the pinned script hash is shipped.
Successful existing regressions were not repeated after changes confined to the
new research cell. No notebooks exist in this test workspace.

Heart remains RED for `release validation FAILED (stage integrate)`; the live
human approved a task-specific development-only override through PR creation.
No merge or release is authorized. Branch validation does not clear Heart's
separate workspace-timeout or manifest-drift readiness reasons.
