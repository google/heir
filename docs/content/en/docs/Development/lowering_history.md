---
title: Lowering history in diagnostics
---

For the torch-linalg pipeline, enable named checkpoints with:

```sh
heir-opt model.mlir --torch-linalg-to-ckks='lowering-history=true' \
  --mlir-print-debuginfo
```

Checkpoints record input to linalg preprocessing, its output, input to layout
assignment, and input to ciphertext conversion. Errors gain notes such as:

```text
model.py:42:1: error: ...
model.py:42:1: note: lowering history: 'arith.maximumf' observed at stage 'after-linalg-preprocessing'
model.py:42:1: note: lowering history: 'linalg.generic' observed at stage 'torch-linalg-input'
```

The particular history depends on which locations each rewrite forwards. A
checkpoint describes an operation observed at a stage, not an assertion that the
preceding pass changed it. Missing checkpoints are not evidence that no rewrite
occurred. This is a first step toward detailed transformation provenance, not a
complete per-rewrite trace.

Selected rewrites also record explicit attribution. ReLU select-to-maximum
canonicalization and unary/constant-binary polynomial approximation currently
use `getLoweringLocation`. Their notes say, for example:

```text
note: lowering history: 'arith.select' lowered to 'arith.maximumf' by --activation-canonicalizations
```

The helper preserves the original location when history is disabled. When adding
it to another rewrite, call it only after the pattern has matched, provide the
actual result operation name, and use the returned location on that new
operation. Stage observations and explicit rewrite events are rendered
differently; the compiler does not infer attribution from a pass merely having
run.

Custom pipelines can place checkpoints around any sequence of transformations:

```sh
heir-opt model.mlir \
  --pass-pipeline='builtin.module(explain-lowering-history,record-lowering-history{stage=imported},activation-canonicalizations,record-lowering-history{stage=after-activation})' \
  --mlir-print-debuginfo -o intermediate.mlir
```

A later compiler process can explain the serialized history by running
`--explain-lowering-history` before its lowering passes. Enable it before passes
that may fail; enabling it after a failure cannot change the earlier diagnostic.
Ordinary successful compilations do not print history notes. The default is
eight history notes per error, followed by an omission note if necessary. Use
`--explain-lowering-history=max-notes=4` to change that limit.

Each checkpoint wraps the existing location with a `NameLoc` for the current
operation and a `FusedLoc` carrying `heir.lowering_stage` metadata. Original
source locations and call stacks remain nested inside. Named boundaries keep
repeated checkpoints distinct, even when their stage names match. Multiple
origins can be represented by fusing these locations; explanation visits each
distinct origin.

Recording is opt-in and adds location storage proportional to the distinct
operation-name/location pairs at each checkpoint. Prefer a few meaningful
stages. The note limit bounds rendered output, not history storage. A rewrite
that drops locations also drops history; a rewrite forwarding one input's
location retains only that origin. Use upstream tagged `--snapshot-op-locations`
as well when you need a file/line reference into the actual intermediate IR.

<!-- mdformat global-off -->
