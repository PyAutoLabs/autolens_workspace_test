# Mesh interpolator numerical audit

Audit for PyAutoArray #603. Production algorithms and defaults are unchanged.
Confirmed defects were filed separately as [#609](https://github.com/PyAutoLabs/PyAutoArray/issues/609)
and [#610](https://github.com/PyAutoLabs/PyAutoArray/issues/610). Four narrow,
strict expected failures retain their numerical oracles; an unexpected pass
fails the suite so the corresponding regression must be promoted after repair.

The permanent library regressions are in
`test_autoarray/inversion/pixelization/interpolator/test_numerics_audit.py`.
The companion `mesh_interpolator_source_recovery.py` is registered in
`smoke_tests.txt`.

## Applicability and fault witnesses

| Family | Linear precision | Boundaries | Supported physical refinement | Synthetic source recovery | Deliberately broken variants rejected |
| --- | --- | --- | --- | --- | --- |
| Adaptive rectangular, rank | Exact in CDF index space | Both axes, exact/ULP/finite integer-cell probes | Tested with fixed, untied, smooth uniform marginals; arbitrary empirical rank inverses have no universal second-order guarantee | Same class audited through its kernel variant | Permuted corner weights, one-row cell assignment, frozen coarse mesh |
| Adaptive rectangular, default kernel | Exact in CDF index space | Both axes; actual CDF clip plateaus at bounding-box extrema, exact/nextafter/finite probes without replacing the transform | Independent Gaussian-mixture CDF root inversion; default production knot count retained | Independent physical-node truth; historical row mirroring fails the same criterion | Weight pairing, plateau-only cell assignment, frozen mesh, quantized inverse table, geometry edge order |
| Rectangular uniform | Exact in physical coordinates | Both axes, exact/nextafter/finite cell crossings | Smooth physical truth | Independent truth; fixed control stays bit-identical across an actual patched adaptive entry point | Weight pairing, shifted cell, frozen mesh, contaminated control, consistent source-node permutation |
| Delaunay | Exact inside hull | Every interior simplex edge, edge-normal exact/nextafter/finite crossings | Smooth truth in supported central region | Independent physical-node truth | Weight pairing, shifted simplex mapping, frozen mesh, source-node permutation |
| DelaunayNN / Sibson | Exact away from diagnosed edge defects | Exact/ULP checks alone miss #610; finite edge-normal probes and independent square symmetry expose it | Smooth supported-region refinement away from diagnosed edge branch | Independent physical-node truth; this passing aggregate check does not certify edge behavior | Weight pairing, cell shift, frozen mesh, diagonal-only coordinates, normalize-then-positive-filter, source-node permutation |
| KNearestNeighbor | No global linear precision or nodal interpolation guarantee; independent nearest-neighbor and Wendland weight oracles, constants | Global neighbor-switch continuity is not assumed | Measured convergence; no global second-order claim | Independent coefficient truth with documented smoothing allowance | Neighbor/weight pairing, deliberately truncated final block, frozen mesh, source-node permutation |
| KNNBarycentric | Exact only where the selected nondegenerate triangle contains the query | Global continuity across nearest-triangle changes is not assumed | Measured convergence; clipped/non-enclosing triangles preclude a global second-order claim | Independent physical-node truth | Triangle/weight pairing, shared final-block defect, frozen mesh, source-node permutation |

Fault witnesses execute inside the tests and require an `AssertionError` from
the unchanged numerical oracle. Even the known-defect regressions exercise a
relevant injected fault before reaching the production failure. Detailed
failure excerpts can be emitted explicitly; ordinary runs stay silent:

```bash
PYAUTO_NUMERICS_AUDIT_EVIDENCE=1 python -m pytest \
  test_autoarray/inversion/pixelization/interpolator/ -q -s -rx
```

## Confirmed findings

**#609, incomplete KNN block.** For 130 points `(i, 0)` and exact queries at
points 128/129, `get_interpolation_weights(..., k_neighbors=3,
radius_scale=1.5)` returns nearest index 127 for both queries, with distances
1 and 2. The expected indices are 128/129 and nearest distances are zero.
`lax.dynamic_slice` clamps the final start while its mask and reported indices
still use the unclamped start. Both KNN classes share this function. The
strict regression checks an independent all-point distance oracle; ordinary
within-block neighbors and kernel weights also have passing independent tests.

**#610, Sibson interior edges.** For square vertices `(-1,-1), (-1,1), (1,-1),
(1,1)`, square symmetry requires four center weights of 1/4. The analytic nodal
field `y*x` must interpolate to zero at the center. Production instead returns
one diagonal with weights 1/2, 1/2 and value 1. At queries `(+-1e-8, 0)`, it
returns a weight sum 1.123877166 and value -1.123877166, with no overflow or
degenerate flag. At displacement .01, the value is approximately zero and the
weights sum to one. The exact-edge substitution and the near-edge
normalize-then-positive-filter behavior are both preserved as strict numerical
regressions. On the 9x9 lattice edge-normal sweep, the maximum discrepancy
between exact and finite-offset values was 0.02680921. No tolerance was loosened
to hide this discrepancy.

## CDF table accuracy

The independent 200-point Gaussian sample uses seed 22. Values below are
maximum forward/inverse round-trip errors in mesh index units over interior
unit-CDF probes. The permanent test uses the actual default for its first
column, which is 64 knots in this snapshot.

| Mesh size | Default 64 knots | 256 knots | 1024 knots |
| --- | ---: | ---: | ---: |
| 16 | 0.002764393669 | 0.000210916214 | 0.000012491579 |
| 32 | 0.018877478111 | 0.001042458641 | 0.000077065088 |
| 64 | 0.101891425555 | 0.006066538508 | 0.000311042291 |

This warrants a separate error-budget-driven knot-scaling investigation.
For this fixture, 256 knots keep even the size-64 mesh below 0.01 index units;
1024 knots keep all three sizes below 0.001. These measurements do not establish
an adequate universal multiplier across source distributions. Select the
allowed index error, then validate a mesh-size-dependent table policy across
representative distributions and its cost before changing defaults.

Physical refinement tests do not replace the production default with a finer
inverse table to manufacture convergence. Kernel truth nodes are independently
root-solved from the defining Gaussian mixture. The clean rank refinement
fixture has fixed, untied, evenly spaced marginals, so its inverse is smooth at
the tested scale. Important bounded limitations found during fixture study:
physical RMS at sizes 8/16/32 was .0819282/.0495717/.0228168 for tied lattices,
and .0246593/.0114359/.0143091 for a fixed lattice with narrow marginal-coordinate clusters after
small jitter. Those empirical-rank inverse transitions do not justify a blanket
second-order or monotonic physical-refinement promise. They are recorded as
limitations, not hidden behind relaxed thresholds.

## Guard and boundary conventions

For adaptive mapping, transformed coordinates lie in `[1, n-2]` and
`flat = (n - index_y)*n + index_x`. Positive-weight support therefore occupies
flat rows **2 through n-1**, columns **1 through n-2**. Row 1 can occur only as a
zero-weight corner at the upper plateau; row 0 has no support. Zero-weight
stencil entries must not be confused with live nodes.

The geometry's CDF edge coordinates
`(n-row-.5)/(n-3)` and `(col-1.5)/(n-3)` agree with those node positions. A
permanent test compares the actual geometry edges against independently
constructed Gaussian-mixture knot coordinates and rejects a shifted edge
ordering. It does not use `areas_transformed` or depend on the geometry repair.

`zeroed_pixels` uses the conventional perimeter. Consequently its overlap with
positive support is the last flat row; it is not identical to the inactive
guard set. Whether that asymmetric inversion boundary condition is intended
remains an unresolved semantics question, not a confirmed defect in this audit.

The original-transform plateau test uses both bounding-box extrema on both
axes and their nextafter neighbors, plus finite displacements. It checks an
independent dense Gaussian-mixture CDF oracle. The interior integer-cell test
separately isolates discretization from the approximate inverse to avoid
pretending a knot lookup can land on an exact integer boundary.

## Independent synthetic source recovery

The 51x51 synthetic image grid has an affine image-to-source ray map. An
asymmetric two-component Gaussian truth supplies the observations analytically.
No observations are generated as a candidate mapping matrix times its own
source. Least-squares coefficients are compared at independently located
physical mesh nodes, rather than forwarded through the possibly wrong basis.
This is an isolated interpolation/source-recovery harness: it does not certify
PSF convolution, nonlinear lens fitting, or regularization choices.

The adaptive kernel mesh has size 16. Truth nodes are located by root-solving
the defining Gaussian-mixture CDF, independently of the production inverse knot
table. Its recovery criterion is Pearson r > .995 and normalized RMS < .07.
Mirrored row weights fail that same criterion. The other families are evaluated
at data-supported interior nodes with r > .98 and RMS < .20; the wider coefficient
allowance recognizes that kernel KNN is a smoothing basis. Their consistent
four-row source-node permutations fit an equally permuted basis but fail the
physical truth criterion.

| Family | Correct Pearson r | Correct normalized RMS | Broken Pearson r | Broken normalized RMS |
| --- | ---: | ---: | ---: | ---: |
| Adaptive kernel | .999529646 | .040116375 | .995909035 | .075801230 |
| Uniform | .999433789 | .042378823 | .045692114 | .897174271 |
| Delaunay | .999383795 | .042320412 | .048097038 | .895955130 |
| Sibson | .999354528 | .044957657 | .031543978 | .903864016 |
| Kernel KNN | .996833898 | .100590551 | .103416382 | .918651813 |
| KNN barycentric | .999445224 | .041967083 | .046176894 | .896898244 |

The uniform control is bit-identical before and during the actual temporary
adaptive entry-point mutation. A contaminated uniform control is also required
to differ, proving that the control check itself is discriminating. The rank
alias of the adaptive class has precision/boundary/refinement coverage; source
recovery in this harness exercises that class through its default kernel variant.

## Validation and limits

- Full PyAutoArray snapshot: **1940 passed, 1 strict expected failure**, 80
  existing warnings, exit 0, 461.98 seconds. This run preceded the final added
  finite-normal/plateau/guard regressions; it did not certify the later-discovered
  Sibson issue. Production files did not change.
- Final complete interpolator suite: **90 passed, 4 strict expected failures**,
  exit 0, 46.74 seconds. The new audit contributes 28 passing cases and four
  expected failures (one for #609, three for #610); 33 deliberately broken
  variant witnesses emitted captured assertion failures.
- Actual Hands smoke runner with workspace `profile_smoke.yaml`: the new
  registered source-recovery entry **passed**, exit 0, 8.27 seconds. Direct
  execution retained the per-family metrics above.
- The local interpreter is `/home/jammy/venv/PyAuto/bin/python`; activation
  imports `autoarray` from the audit worktree. Python 3.12/3.13 CI checks remain
  required; this local run does not claim both matrix legs.

Full local logs, explicit exit files, mutation failure excerpts, exact defect
reproductions, and the structured smoke report are retained under the bundle's
`scratch/numerics-audit/`. No production fixes, commits, PR creation, registry
changes, or merges were performed by the audit worker.
