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

## Model reduction

Two Orbit techniques shrink the ILP before it is solved; both are exact with
respect to the objective because op costs are linear in the execution level.

**Compression** (`compress`, default true) groups ops so each group shares one
set of ILP variables and one management decision per merged site:

- *Addition squashing*: a maximal tree of additions whose interior fanout stays
  inside the tree executes at a single (level, scale) and needs one management
  decision after its final addition. Note: Addition-tree squashing applies only
  to additions. Rotate-and-sum trees are deliberately not squashed; equivalent
  rotations are reduced by structural merging instead.
- *Structural merging* (`auto_compress`): via iterative label refinement, ops
  with the same depth, op class, and (at the fixpoint) the same producer classes
  share one variable set. Each merged op still decodes its own management ops,
  and the objective charges the group once per member, so the compressed model's
  optimum equals the original optimum restricted to symmetric solutions.

**SISO partitioning** (`partition-min-size`, default 100) cuts the circuit where
exactly one value is live across the cut. Each partition is solved independently
for every reachable boundary input state and every enumerated boundary output
level, and a dynamic program stitches the per-partition solutions. At most one
scale per boundary level survives between partitions, so the stitched placement
is a high-quality heuristic rather than provably optimal — matching Orbit's
implementation.

## Solver configuration

The ILP is solved to a fixed 1% relative optimality gap with no time limit,
matching Orbit's solver configuration. Proving full optimality often dominates
solve time on large instances while improving the objective by less than
measurement noise in the profiled cost models.

<!-- mdformat global-off -->
