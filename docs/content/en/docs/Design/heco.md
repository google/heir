---
title: HECO SIMD Optimizations
weight: 9
---

HEIR includes a SIMD (Single Instruction, Multiple Data) optimizer which is
designed to exploit the restricted SIMD parallelism most (Ring-LWE-based) FHE
schemes support (also commonly known as "packing" or "batching"). Specifically,
HEIR incorporates the "automated batching" optimizations (among many other
things) from the [HECO](https://github.com/MarbleHE/HECO) compiler. The
following will assume basic familiarity with the FHE SIMD paradigm and the
high-level goals of the optimization, and we refer to the associated HECO
[paper](https://www.usenix.org/system/files/usenixsecurity23-viand.pdf),
[slides](https://www.usenix.org/system/files/sec23_slides_viand.pdf),
[talk](https://www.youtube.com/watch?v=SP3C6gLWIS4) and additional resources on
the
[Usenix'23 website](https://www.usenix.org/conference/usenixsecurity23/presentation/viand)
for an introduction to the topic. This documentation will mostly focus on
describing how the optimization is realized in HEIR (which differs somewhat from
the original implementation) and how the optimization is intended to be used in
an overall end-to-end compilation pipeline.

## Representing FHE SIMD Operations

Following the design principle of maintaining programs in standard MLIR dialects
as long as possible (cf. the design rationale behind the
[Secret Dialect](https://heir.dev/docs/design/secret/)), HEIR uses the MLIR
[`tensor` dialect](https://mlir.llvm.org/docs/Dialects/TensorOps/) and
[ElementwiseMappable](https://mlir.llvm.org/docs/Traits/#elementwisemappable)
operations from the MLIR
[`arith` dialect](https://mlir.llvm.org/docs/Dialects/ArithOps/) to represent HE
SIMD operations.

We do introduce the HEIR-specific
[`tensor_ext.rotate`](https://heir.dev/docs/dialects/tensorext/#tensor_extrotate-heirtensor_extrotateop)
operation, which represents a cyclical left-rotation of a tensor. The SIMD
vectorizer only rotates one-dimensional tensors (it flattens multi-dimensional
tensors in row-major order first), and the semantics of this operation on
multi-dimensional tensors are not (currently) defined.

For example, the common "rotate-and-reduce" pattern which results in each
element containing the sum/product/etc of the original vector can be expressed
as:

```llvm
%tensor = tensor.from_elements %i1, %i2, %i3, %i4, %i5, %i6, %i7, %i8 : tensor<8xi16>
%0 = tensor_ext.rotate %tensor, %c4 : tensor<8xi16>, index
%1 = arith.addi %tensor, %0 : tensor<8xi16>
%2 = tensor_ext.rotate %1, %c2 : tensor<8xi16>, index
%3 = arith.addi %1, %2 : tensor<8xi16>
%4 = tensor_ext.rotate %3, %c1 : tensor<8xi16>, index
%5 = arith.addi %3, %4 : tensor<8xi16>
```

The `%cN` and `%iN`, which are defined as `%cN = arith.constant N : index` and
`%iN = arith.constant N : i16`, respectively, have been omitted for readability.

## Intended Usage

The `-heco-simd-vectorizer` pipeline transforms a program consisting of loops
and index-based accesses into tensors (e.g., `tensor.extract` and
`tensor.insert`) into one consisting of SIMD operations (including rotations) on
entire tensors. While its implementation does not depend on any FHE-specific
details or even the Secret dialect, this transformation is likely only useful
when lowering a high-level program to an arithmetic-circuit-based FHE scheme
(e.g., B/FV, BGV, or CKKS). The `--mlir-to-bgv --scheme-to-openfhe` pipeline
demonstrates the intended flow: augmenting a high-level program with `secret`
annotations, then applying the SIMD optimization (and any other high-level
optimizations) before lowering to BGV operations and then exiting to OpenFHE.

Multi-dimensional tensors are treated as their row-major flattening, i.e., a
tensor with `n` elements occupies `n` slots.

Since the optimization is based on heuristics, the resulting program might not
be optimal or could even be worse than a trivial realization that does not use
ciphertext packing. However, well-structured programs generally lower to
reasonable batched solutions, even if they do not achieve optimal batching
layouts. For common operations such as matrix-vector or matrix-matrix
multiplications, state-of-the-art approaches require advanced packing schemes
that might map elements into the ciphertext vector in non-trivial ways (e.g.,
diagonal-major and/or replicated). The SIMD vectorizer keeps input tensors in
their row-major arrangement and only moves elements (with rotations and masks)
to align them with the slots a loop nest computes in. The one exception is the
reindexing of reduction loops described below, which yields the diagonal method
for matrix-vector products with a plaintext matrix; it does not search for other
packings.

Note, that the SIMD batching optimization is different from, and significantly
more complex than, the Straight Line Vectorizer (`-straight-line-vectorize`
pass), which simply groups
[ElementwiseMappable](https://mlir.llvm.org/docs/Traits/#elementwisemappable)
operations that agree in operation name and operand/result types into
vectorized/tensorized versions.

## Implementation

Below, we give a brief overview over the implementation, with the goal of both
improving maintainability/extensibility of the SIMD vectorizer and allowing
advanced users to better understand why a certain program is transformed in the
way it is.

### Components

The `-heco-simd-vectorizer` pipeline consists of the
[`-simd-vectorize-loops`](https://heir.dev/docs/passes/tensorextpasses/#-simd-vectorize-loops)
pass, followed by
[`-canonicalize`](https://mlir.llvm.org/docs/Passes/#-canonicalize) and
[`-cse`](https://mlir.llvm.org/docs/Passes/#-cse) to apply the rotation-specific
canonicalization patterns described in
[TensorExt Canonicalization](#tensorext-canonicalization).

### Loop Vectorization

Loop nests that compute a tensor one element at a time, or reduce a tensor's
elements into a scalar, are rewritten into SIMD operations and rotations
directly, without unrolling the loops. The pass models each nest's iteration
domain as an integer set and uses the index of the `tensor.insert` (or of a
`tensor.extract`, see below) to assign each iteration a slot. This is the
"target slot" of the HECO batching optimization, expressed as an affine function
of the loop induction variables instead of being chosen separately for each
operation of the unrolled program.

Every `tensor.extract` whose index has a constant offset from that slot becomes
a rotation of the source tensor by that offset. For example, the loop computing
`t2[i] = t0[i - 1] + t1[i + 1]` becomes

```llvm
%0 = tensor_ext.rotate %t0, %c31 : tensor<32xi16>, index
%1 = tensor_ext.rotate %t1, %c1 : tensor<32xi16>, index
%2 = arith.addi %0, %1 : tensor<32xi16>
```

i.e., rotating `t0` down by one (31 = -1 (mod 32)) and `t1` up by one so that
the elements at indices `i - 1` and `i + 1` are aligned with slot `i`. In the
unrolled HECO formulation, each iteration produces the same rotations, which
common subexpression elimination then merges; analyzing the loop directly
produces the merged form at once.

An extraction from a plaintext tensor whose offset is not constant becomes a
plaintext vector that gathers each iteration's element into its slot. An
extraction from a secret tensor whose offset is not constant is relaid out with
one rotation and mask per distinct offset, which corresponds to the rotations
the unrolled HECO formulation would emit for these iterations.

Loops nested inside a vectorized nest that reduce into a scalar (e.g., the taps
of a convolution kernel) remain loops over whole tensors, and are replaced by a
`tensor_ext.rotate_and_reduce` when they sum rotations of a single tensor,
optionally weighted by plaintexts. When the offset of an extraction from a
secret tensor inside such a loop depends on the slot, the loop may be reindexed
so that it does not: for a matrix-vector product with a plaintext matrix,
iterating over the diagonals of the matrix instead of its columns gives the
diagonal (Halevi-Shoup) method.

If no slot map derived from the insertion index works, the pass instead takes
the index of an extraction as the slot map, which also vectorizes the reduction
loops it depends on. A reduction over such a loop becomes a
`tensor_ext.rotate_and_reduce` over the slots of its iterations, and the results
are moved to the output slots at the end. For example, the row sums of a 4x4
matrix are computed in the 16 slots of the matrix with two rotations, and then
moved to the four output slots with three more.

### TensorExt Canonicalization

The
[TensorExt (`tensor_ext`) Dialect](https://heir.dev/docs/dialects/tensorext/)
includes a series of canonicalization rules that are essential to making
automatically generated rotation code efficient:

- Rotation by zero: `rotate %t, 0` folds away to `%t`

- Cyclical wraparound: `rotate %t, k` for $k > t.size$ can be simplified to
  `rotate %t, (k mod t.size)`

- Sequential rotation: `%0 = rotate %t, k` followed by `%1 = rotate %0, l` is
  simplified to `rotate %t (k+l)`

- Extraction: `%0 = rotate %t, k` followed by `%1 = tensor.extract %0[l]` is
  simplified to `tensor.extract %t[k+l]`

- Binary Arithmetic Ops: where both operands to a binary `arith` operation are
  rotations by the same amount, the rotation can be performed only once, on the
  result. For Example,

  ```llvm
  %0 = rotate %t1, k
  %1 = rotate %t2, k
  %2 = arith.add %0, %1
  ```

  can be simplified to

  ```llvm
  %0 = arith.add %t1, %t2
  %1 = rotate %0, k
  ```

- *Sandwiched* Binary Arithmetic Ops: If a rotation follows a binary `arith`
  operation which has rotation as its operands, the post-arith operation can be
  moved forward. For example,

  ```llvm
  %0 = rotate %t1, x
  %1 = rotate %t2, y
  %2 = arith.add %0, %1
  %3 = rotate %2, z
  ```

  can be simplified to

  ```llvm
  %0 = rotate %t1, x + z
  %1 = rotate %t2, y + z
  %2 = arith.add %0, %1
  ```

- Single-Use Arithmetic Ops: Finally, there is a pair of rules that do not
  eliminate rotations, but move rotations up in the IR, which can help in
  exposing further canonicalization and/or CSE opportunities. These only apply
  to `arith` operations with a single use, as they might otherwise increase the
  total number of rotations. For example,

  ```llvm
  %0 = rotate %t1, k
  %2 = arith.add %0, %t2
  %1 = rotate %2, l
  ```

  can be equivalently rewritten as

  ```llvm
  %0 = rotate %t1, (k+l)
  %1 = rotate %t2, l
  %2 = arith.add %0, %1
  ```

  and a similar pattern exists for situations where the rotation is the rhs
  operand of the arithmetic operation.

Note that the index computations in the patterns above (e.g., `k+l`,
`k mod t.size` are realized via emitting `arith` operations. However, for
constant/compile-time-known indices, these will be subsequently constant-folded
away by the canonicalization pass.

<!-- mdformat global-off -->
