# ILP Bootstrap Placement

## Cost Model JSON

`--ilp-bootstrap-placement="orbit-cost-model=PATH"` loads an optional JSON cost
model and uses it to override the `bootstrap-cost` and `rescale-cost` pass
options. The units of latency are in microseconds. If level-dependent costs are
needed, the next step would be to extend this schema and the ILP objective.

Required fields:

- `latencyTable.bootstrap`: numeric latency samples for one bootstrap chosen by
  the ILP.
- `latencyTable.rescale`: numeric latency samples for one unit of rescale,
  modreduce, or level-reduce management chosen by the ILP.

<!-- mdformat global-off -->
