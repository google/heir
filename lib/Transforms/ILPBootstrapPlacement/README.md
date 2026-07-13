# ILP Bootstrap Placement

## Cost Model JSON

`--ilp-bootstrap-placement="orbit-cost-model=PATH"` loads an optional JSON cost
model. The units of latency are in microseconds.

Every key under `latencyTable` maps to a per-level latency array, where index
`i` holds the latency at level `i + 1`. All keys are required; loading fails if
any is missing:

- `bootstrap`: the positive-sample average is the constant bootstrap cost in the
  objective (levels below the bootstrappable range may be recorded as zero).
  Overrides the `bootstrap-cost` option.
- `rescale`: the per-level maximum is the constant cost of one unit of rescale,
  modreduce, or level-reduce management chosen by the ILP. Overrides the
  `rescale-cost` option.
- `addCtCt`, `addCtPt`, `mulCtCt`, `mulCtPt`, `rotate`, `negate`: each array is
  least-squares fitted to `cost(level) = slope * level + intercept`, and each
  tracked op is charged its fitted cost at its ILP-chosen input level in the
  objective. `CtCt` is the ciphertext-ciphertext variant of a binary op, `CtPt`
  the ciphertext-plaintext variant.

## Solver configuration

The ILP is solved to a fixed 1% relative optimality gap with no time limit,
matching Orbit's solver configuration. Proving full optimality often dominates
solve time on large instances while improving the objective by less than
measurement noise in the profiled cost models.

<!-- mdformat global-off -->
