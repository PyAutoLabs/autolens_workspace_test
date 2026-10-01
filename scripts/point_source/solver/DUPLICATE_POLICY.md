# PointSolver duplicate-image policy — cluster arc phase 1c

Task: [autolens_workspace_test#331](https://github.com/PyAutoLabs/autolens_workspace_test/issues/331).
This is numerical research, not a production solver change or a declaration that
the parent PointSolver health phase is complete.

## Decision

**NO-GO for promoting any of the three tested grouping rules to production.**
Distance-only grouping, shared-edge grouping and bounded root polishing all
remove the duplicated representatives in the centred quad. That success does
not establish a safe general rule. The uncapped NumPy controls already fail
near the cusp: many centroids are far from any true image despite small
source-plane residuals. Root polishing also accepts inaccurate positions near
the critical curve. These are image-plane accuracy/identity problems independent
of capacity. JAX additionally truncates containing-triangle selection.

Two distinct contracts need attention: image-plane precision/identity near
critical curves, and observable overflow under eager execution, JIT and vmap.
A follow-up on overflow must not claim to repair the uncapped NumPy failures.
Raising the cap alone is not a completeness guarantee. Revisit
duplicate-image handling with a conditioning-aware image-plane accuracy
criterion; the uncapped NumPy rows already provide evidence for that work.
Do not begin arc phase 2 on the strength of this report.

## Reproduce

From the workspace root, with the pinned source libraries in the evidence on
`PYTHONPATH`, CPU JAX and x64 enabled:

```bash
python scripts/point_source/solver/duplicate_policy.py
python scripts/point_source/solver/duplicate_policy.py --summarize
```

The script rejects the small-dataset shortcut. It writes
`duplicate_policy_evidence.json`, including raw positions, residuals, parity,
uncapped per-step counts, final triangles, candidate-policy outputs and library
SHAs/package versions. `--summarize` checks a complete saved matrix and derives
its summary without solving again or rewriting the file. The evidence pins
the generating script by SHA256; summary validation rejects a different revision. The full research sweep is opt-in, outside
the smoke allowlist; its successful exit validates the evidence contract, not
the correctness of any candidate grouping policy.

## Experiment

- Reuse the phase-1a SIS plus external-shear quad and SIE mass model. The quad
  is tested at source `(0, 0)` and offsets `±(1e-5, 2e-5)` arcsec, initial scales
  `0.2` and `0.05`, and requested precisions `1e-3` and `1e-4` arcsec.
- For the SIE, locate the symmetry-axis cusp from the critical condition,
  then move inward by fractional distances `1e-3`, `1e-5` and `1e-7`. Test
  initial scale `0.2` at both precisions. The original phase-1a source is not
  rerun here: these are new source coordinates on the same mass model.
- Run NumPy, eager JAX and scalar JIT; additionally vmap the three sources of
  each mass model at fine precision. Production `solve` is called separately
  from the instrumented step traversal: returning intermediates can affect
  compiled arithmetic at a boundary. Geometry interpretation is withheld if
  its centroids do not match the production output.
- References eliminate radius from the isothermal lens equation and bracket
  its angular roots, independent of triangle candidates. Angular grids of
  4096 and 8192 points, supplemented by cusp-focused angles, must recover the
  same four roots and satisfy the full lens equation. This oracle is specific
  to these two mass models; it is not a general cluster-lens oracle.

## Tested rules

All rules select the lexicographically first raw centroid as representative,
making output independent of input-row order. They do not replace production
centroids with polished roots. The recorded positional check requires four
representatives and assignment error no larger than twice requested precision.

1. **Distance:** group within twice requested precision, as in the historical
   audit diagnostic. A separate negative control applies this rule directly to
   the independent true roots, so failure cannot be blamed on triangle output.
2. **Shared edge:** group representatives of triangles sharing two vertices
   within `1e-10` arcsec, when geometry correspondence is verified. This tests
   a greedy shared-edge rule, not every possible topology algorithm.
3. **Root identity:** polish each candidate with the full lens equation;
   require residual below `1e-10`, movement no greater than twice requested
   precision and unchanged parity. Group valid same-parity candidates whose
   polished positions differ by less than `1e-7` arcsec. Keep failed or
   ambiguous candidates separate. This is a host-side diagnostic heuristic,
   not a certified uniqueness test or a JAX-compatible implementation.

## Measurements

All **54 scalar rows and six vmap controls completed**. The final script was
rerun end to end after independent-review corrections. Its SHA256 is retained
in the JSON, and the completion log explicitly records the successful
before/after library/environment and script-provenance checks.

| Quad control | NumPy rows | Eager JAX rows | JIT rows | All three grouping rules |
|---|---:|---:|---:|---|
| Centred, scale 0.2, either precision | 4 | 4 | 4 | Four images retained |
| Centred, scale 0.05, either precision | 6 | 6 | 4 | Four representatives recovered |
| Either source offset, either scale/precision | 4 | 4 | 4 | Four images retained |

All 36 quad rows meet the four-image positional criterion after each rule.
The worst assignment error is `0.782 × requested precision`. Across all six
reference cases, the largest angular-grid convergence difference is
`4.73e-13` arcsec and the largest lens-equation residual is `4.71e-16` arcsec.
None reaches the capacity. The duplicate rows correspond to adjacent
triangles straddling the same analytic image; successful grouping of those
rows alone therefore gives no evidence about genuinely close images.

| Inward cusp fraction | Precision | NumPy count | Eager JAX / JIT count | Max uncapped NumPy containing count |
|---|---:|---:|---:|---:|
| 1e-3 | 1e-3 | 10 | 7 / 7 | 31 |
| 1e-3 | 1e-4 | 4 | 3 / 3 | 31 |
| 1e-5 | 1e-3 | 40 | 19 / 19 | 41 |
| 1e-5 | 1e-4 | 86 | 13 / 13 | 87 |
| 1e-7 | 1e-3 | 40 | 19 / 19 | 41 |
| 1e-7 | 1e-4 | 82 | 15 / 15 | 83 |

Every near-cusp reference has four independent roots. Their closest pair
separations are respectively `0.0706273`, `0.00706256` and `0.000706240`
arcsec. All 18 near-cusp scalar rows exceed the JAX capacity; only the 12
JAX rows actually truncate. NumPy remains dynamically sized and untruncated.
At fraction `1e-3`, precision `1e-4`, the positive-y/positive-x root near
`(1.23972, 1.13986)` is absent from both eager and compiled JAX outputs but
present in NumPy. The new `reference_coverage.nearest_candidate_distances`
records coverage per true image instead of inferring completeness from counts.
For example, the same root is about `0.068` arcsec from the nearest JAX candidate
at precision `1e-3`, even though seven candidates are returned. Fine controls
closer to the cusp also lack candidates close to multiple true images. These
are observed coverage failures after capped selection, not cap-size A/B
experiments proving truncation is their sole cause.

Each rule meets the four-image positional criterion in **37/54** scalar rows:
the 36 quad rows plus the fine NumPy `1e-3` cusp row. The untruncated NumPy
failures are retained in the summary's resolved-failure list. They establish
that making JAX overflow observable cannot by itself settle image identity.

| Untruncated NumPy control | Candidates | Median distance to nearest true root | Maximum | Candidates within 2×precision |
|---|---:|---:|---:|---:|
| cusp 1e-5, precision 1e-4 | 86 | 9.1e-4 | 3.5e-3 | 10 |
| cusp 1e-7, precision 1e-4 | 82 | 1.7e-3 | 4.9e-3 | 7 |
| cusp 1e-5, precision 1e-3 | 40 | 3.4e-3 | 1.2e-2 | 12 |

These distributions are not simply multiple representatives of accurately
located images. Small source-plane residuals can accompany large image-plane
offsets near a critical curve. A grouping rule that only selects existing
centroids is not a remedy for that accuracy problem.

The root-identity heuristic has a separate counterexample: for uncapped NumPy,
cusp `1e-7`, precision `1e-4`, it marks eight polished candidates valid for four
true images. Four are still roughly `5.9e-5`–`2.7e-4` arcsec from a true root,
although their residuals are around `1e-12`. A fixed residual threshold is not
an image-plane error certificate for a near-singular Jacobian. Neither the
polisher's success nor its `valid` flag establishes root identity; a future
policy must account for conditioning and verify image-plane accuracy.

The distance-only negative control collapses the **four true roots to two**
at cusp fraction `1e-7` and precision `1e-3`. These are distinct independently
resolved roots, although their separation is below the requested triangle
resolution. The proposed production grouping must therefore expose unresolved
multiplicity rather than interpreting a precision-sized radius as identity.

All six vmap outputs match their scalar JIT outputs exactly, including the
three-image missing-root case. Batch parity alone cannot establish physical
completeness. Root-identity representative selection also passes the reversed
input-order check for every scalar and vmap row.

## Production contract and limits

- A nearby centroid is not evidence of identical physical-image identity.
  Even shared edges and low residuals require care near a critical curve.
  Unresolved pairs should be explicitly identified or further refined, not
  silently collapsed. This experiment does not define a certified resolution
  or root-uniqueness threshold for general lenses.
- `exceeds_jax_capacity` means an uncapped count above 20, on either backend.
  `selection_truncated` is true only for JAX when that happens. NumPy has no
  such cap and its rows remain eligible for the resolved-failure summary.
  JAX-truncated rows are inconclusive for a clean cross-backend identity
  comparison; subsequent steps can diverge after the first truncation.
- A production repair must preserve static padded JAX output and distinguish
  overflow/ambiguity from a legitimate zero-image result. The natural source
  targets are PyAutoArray `ArrayTriangles.containing_indices` and PyAutoLens
  `AbstractSolver.steps` / `PointSolver._solve_array`; downstream callers need
  an explicit policy for invalid results. No API choice is approved here.
- A future identity rule belongs before the implicit-differentiation wrapper
  returns the chosen images. It must preserve representative identity across
  backend and batch paths; image-count changes remain seams. The new sweep
  measures forward solutions with an unregistered tracer, not gradients or
  the registered custom-JVP path. Existing padding coverage exercises registered
  tracers separately and cannot validate a future identity rule's gradients.
- Only single-plane simulated fixtures, CPU and fp64 were studied. Neither
  arbitrary cluster completeness nor GPU, fp32 or multi-plane behavior is
  established. The finite angular reference grid and numerical residual checks
  are convergence evidence, not interval proofs of root uniqueness.

## Validation

- Research matrix: final-revision rerun of 54 scalar rows and six vmap controls;
  saved-matrix validation and the analytic-quad cross-check pass. Summary validation
  leaves the evidence SHA256 unchanged.
- `padding_backend.py`: all 12 constructor/backend/tracer cases pass.
- `scripts/point_source/jax_likelihood/image_plane.py`: passes, including the
  registered fit/JIT round trip and the pinned likelihood check.
- Full `.github/scripts/run_smoke.py`: **32 passed, zero failed**, local CPU.
  Machine report: `test-results/autolens_workspace_test__scripts__script.json`.
- Black, Python compilation, JSON/report-total and diff-whitespace checks pass.
  Initial independent Fable review returned FINDINGS. Corrections separate
  NumPy accuracy failures from JAX truncation, expose the root-polishing
  counterexample and per-root coverage, make summary validation read-only,
  and rerun the final script. Independent verdicts are recorded on #331.
  An earlier rerun was discarded after a shared activation symlink pointed to
  a removed task bundle; the successful rerun uses a task-owned activation file.
- Runtime logs: task-bundle `scratch/{duplicate-policy-final-stable,padding,image-plane,smoke}.log`.
  No notebooks exist in this test workspace; no notebook regeneration target.

These checks validate the research artifact and existing workspace behavior;
they do not turn the policy no-go into a production-correctness pass. No
production libraries were edited. At handoff, the feature remains local and
uncommitted pending the separate development-shipping Heart gate.
