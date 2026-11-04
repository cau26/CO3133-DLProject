# Calibration v2: disjoint components and an explicit review policy

The new procedure keeps the original assessment set unchanged. It improves how
the existing calibration data are used; it does not create fresh independent
data or erase the fact that the earlier calibration results were inspected.

## Procedure fixed before inference

`partition_calibration(records, config)` operates only on calibration metadata.
It joins records that share a source group or an available near-duplicate
evaluation component. A deterministic seeded hash assigns approximately half of
the resulting components to temperature fitting and the other half to threshold
selection. Both kinds of group IDs remain disjoint across these partitions.

The temperature fits NLL using all crops in its own partition. Threshold fitting
uses exactly one representative from each threshold component. Its representative
is the row with the smallest seeded hash of its stable annotation identity,
chosen before any logits or labels are inspected. This avoids counting several
correlated crops from one source/duplicate component as independent trials.

The default policy evaluates the predeclared grid `0.00, 0.01, ..., 1.00`. At each
threshold, accepted representatives give an error count and an accepted count.
An exact one-sided Clopper-Pearson upper error bound uses significance
`delta / 101`, with default `delta=0.05`. Bonferroni correction makes these bounds
simultaneous over the entire grid, so choosing an eligible threshold after seeing
all grid results does not ignore the threshold search. Among thresholds with at
least30 accepted components and an upper error bound at most0.05, the procedure
chooses maximum representative coverage, breaking coverage ties toward the lower
threshold.

If no threshold qualifies, the active policy is **review_all**. Every prediction,
including a numerically exact confidence of1.0, requires human review. The module
also records the best empirical95% grid result as an inactive diagnostic. It never
silently activates that empirical policy or lowers the target to force a pass.

## Meaning and limits of the bound

The target population is an equally weighted, metadata-selected representative
of an independent source/duplicate component. It is **not** the crop-weighted
accuracy of all images, a class-conditional guarantee, or the accuracy of arbitrary
user photos. The binomial interpretation requires a fixed model independent of
threshold labels, independent identically distributed representatives, and
future data from the same population. Hashes and source separation do not prove
these assumptions, and historical inspection of the calibration data remains a
limitation. The result therefore says `target_bound_satisfied_under_assumptions`,
never an unconditional accuracy guarantee.

Study completion and technical release readiness are separate from whether a
95%target can be supported. A technically valid review-all service may still show
class probabilities while requiring review for every case. The serving code must
honor the explicit policy mode; using a threshold of1.0 alone is insufficient.

Distinct hashes bind the temperature subset and the representative threshold
subset. The complete metadata partition, grid, representative rule, numerical
bounds and assumptions are saved before assessment. Subsequent assessment is
descriptive/regression evidence; it never changes the temperature or policy.

## APIs

- `partition_calibration(records, config=None)` returns JSON-serializable local
  row indices, metadata/config/subset hashes and component counts.
- `fit_calibration_v2(logits, targets, records, config=None, partition=...)`
  validates the frozen partition and returns fitting evidence plus a
  release-compatible `binding` with `policy.mode`.
- `review_mask(probabilities, binding)` is the reference decision rule. True
  means human review is required.

## Primary references

Guo et al., [On Calibration of Modern Neural Networks](https://proceedings.mlr.press/v70/guo17a.html),
ICML2017, motivates scalar temperature fitting. Geifman and El-Yaniv,
[Selective Classification for Deep Neural Networks](https://papers.nips.cc/paper/2017/file/4a8423d5e91fda00bb7e46540e2b0cf1-Paper.pdf),
NeurIPS2017, develops binomial-error control for selective classification.
Angelopoulos et al., [Learn then Test](https://arxiv.org/abs/2110.01052), frames
finite-set risk control as multiple testing. This implementation uses the simple
fixed-grid exact-binomial/Bonferroni construction, not a claim that every method
from those papers has been reproduced.
