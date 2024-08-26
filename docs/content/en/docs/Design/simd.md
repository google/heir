---
title: SIMD Optimizations
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
operation, which represents a cyclical left-rotation of a tensor. Note that, as
the current SIMD vectorizer only supports one-dimensional tensors, the semantics
of this operation on multi-dimensional tensors are not (currently) defined.

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

The `-heir-simd-vectorizer` pipeline transforms a program consisting of loops
and index-based accesses into tensors (e.g., `tensor.extract` and
`tensor.insert`) into one consisting of SIMD operations (including rotations) on
entire tensors. While its implementation does not depend on any FHE-specific
details or even the Secret dialect, this transformation is likely only useful
when lowering a high-level program to an arithmetic-circuit-based FHE scheme
(e.g., B/FV, BGV, or CKKS). The `-mlir-to-openfhe-bgv` pipeline demonstrates the
intended flow: augmenting a high-level program with `secret` annotations, then
applying the SIMD optimization (and any other high-level optimizations) before
lowering to BGV operations and then exiting to OpenFHE.

> **Warning** The current SIMD vectorizer pipeline supports only one-dimensional
> tensors. As a workaround, one could reshape all multi-dimensional tensors into
> one-dimensional tensors, but MLIR/HEIR currently do not provide a pass to
> automate this process.

Since the optimization is based on heuristics, the resulting program might not
be optimal or could even be worse than a trivial realization that does not use
ciphertext packing. However, well-structured programs generally lower to
reasonable batched solutions, even if they do not achieve optimal batching
layouts. For common operations such as matrix-vector or matrix-matrix
multiplications, state-of-the-art approaches require advanced packing schemes
that might map elements into the ciphertext vector in non-trivial ways (e.g.,
diagonal-major and/or replicated). The current SIMD vectorizer will never change
the arrangement of elements inside an input tensor and therefore cannot produce
the optimal approaches for these operations.

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

The `-heir-simd-vectorizer` pipeline uses a combination of standard MLIR passes
([`-canonicalize`](https://mlir.llvm.org/docs/Passes/#-canonicalize),
[`-cse`](https://mlir.llvm.org/docs/Passes/#-cse),
[`-sccp`](https://mlir.llvm.org/docs/Passes/#-sccp)) and custom HEIR passes.
Some of these
([`-apply-folders`](https://heir.dev/docs/passes/applyfolderspasses/#-apply-folders),
[`-full-loop-unroll`](https://heir.dev/docs/passes/fullloopunrollpasses/#-full-loop-unroll))
might have applications outside the SIMD optimization, while others
([`-insert-rotate`](https://heir.dev/docs/passes/tensorextpasses/#-insert-rotate),
[`-collapse-insertion-chains`](https://heir.dev/docs/passes/tensorextpasses/#-collapse-insertion-chains)
and
[`-rotate-and-reduce`](https://heir.dev/docs/passes/tensorextpasses/#-rotate-and-reduce))
are very specific to the FHE SIMD optimization. In addition, the passes make use
of the `RotationAnalysis` and `TargetSlotAnalysis` analyses.

### High-Level Flow

- **Loop Unrolling** (`-full-loop-unroll`): The implementation currently begins
  by unrolling all loops in the program to simplify the later passes. See
  [#589](https://github.com/google/heir/issues/589) for a discussion on how this
  could be avoided.

- **Canonicalization** (`-apply-folders -canonicalize`): As the
  rotation-specific passes are very strict about the structure of the IR they
  operate on, we must first simplify away things such as tensors of constant
  values. For performance reasons (c.f. comments in the
  `heirSIMDVectorizerPipelineBuilder` function in `heir-opt.cpp`), this must be
  done by first applying
  [folds](https://mlir.llvm.org/docs/Canonicalization/#canonicalizing-with-the-fold-method)
  before applying the full
  [canonicalization](https://mlir.llvm.org/docs/Canonicalization/).

- **Main SIMD Rewrite** (`-insert-rotate -cse -canonicalize -cse`): This pass
  rewrites arithmetic operations over `tensor.extract`-ed operands into SIMD
  operations over the entire tensor, rotating the (full-tensor) operands so that
  the correct elements interact. For example, it will rewrite the following
  snippet (which computes `t2[4] = t0[3] + t1[5]`)

  ```llvm
  %0 = tensor.extract %t0[%c3] : tensor<32xi16>
  %1 = tensor.extract %t1[%c5] : tensor<32xi16>
  %2 = arith.addi %0, %1 : i16
  %3 = tensor.insert %2 into %t2[%c4] : tensor<32xi16>
  ```

  to

  ```llvm
  %0 = tensor_ext.rotate %t0, %c31 : tensor<32xi16>, index
  %1 = tensor_ext.rotate %t1, %c1 : tensor<32xi16>, index
  %2 = arith.addi %0, %1 : tensor<32xi16>
  ```

  i.e., rotating `t0` down by one (31 = -1 (mod 32)) and `t1` up by one to bring
  the elements at index 3 and 5, respectively, to the "target" index 4. The pass
  uses the `TargetSlotAnalysis` to identify the appropriate target index (or
  ciphertext "slot" in FHE-speak). See [Insert Rotate Pass](#insert-rotate-pass)
  below for more details. This pass is roughly equivalent to the `-batching`
  pass in the original HECO implementation.

  Doing this rewrite by itself does not represent an optimization, but if we
  consider what happens to the corresponding code for other indices (e.g.,
  `t2[5] = t0[4] + t1[6]`), we see that the pass transforms expressions with the
  same relative index offsets into the exact same set of rotations/SIMD
  operations, so the following
  [Common Subexpression Elimination (CSE)](https://en.wikipedia.org/wiki/Common_subexpression_elimination)
  will remove redundant computations. We apply CSE twice, once directly (which
  creates new opportunities for canonicalization and folding) and then again
  after that canonicalization. See
  [TensorExt Canonicalization](#tensorext-canonicalization) for a description of
  the rotation-specific canonocalization patterns).

- **Cleanup of Redundant Insert/Extract**
  (`-collapse-insertion-chains -sccp -canonicalize -cse`): Because the
  `-insert-rotate` pass maintains the consistency of the IR, it emits a
  `tensor.extract` operation after the SIMD operation and uses that to replace
  the original operation (which is valid, as both produce the desired scalar
  result). As a consequence, the generated code for the snippet above is
  actually trailed by a (redundant) extract/insert:

  ```llvm
  %extracted = tensor.extract %2[%c4] : tensor<32xi16>
  %inserted = tensor.insert %extracted into %t2[%c4] : tensor<32xi16>
  ```

  In real code, this might generate a long series of such extraction/insertion
  operations, all extracting from the same (due to CSE) tensor and inserting
  into the same output tensor. Therefore, the `-collapse-insertion-chains` pass
  searches for such chains over entire tensors and collapses them. It supports
  not just chains where the indices match perfectly, but any chain where the
  relative offset is consistent across the tensor, issuing a rotation to realize
  the offset (if the offset is zero, the canonicalization will remove the
  redundant rotation). Note, that in HECO, insertion/extraction is handled
  differently, as HECO features a `combine` operation modelling not just simple
  insertions (`combine(%t0#j, %t1)`) but also more complex operations over
  slices of tensors (`combine(%t0#[i,j], %t1)`). As a result, the equivalent
  pass in HECO (`-combine-simplify`) instead joins different `combine`
  operations, and a later fold removes `combines` that replace the entire target
  tensor. See issue [#512](https://github.com/google/heir/issues/512) for a
  discussion on why the `combine` operation is a more powerful framework and
  what would be necessary to port it to HEIR.

- **Applying Rotate-and-Reduce Patterns**
  (`-rotate-and-reduce -sccp -canonicalize -cse`): The rotate and reduce pattern
  (see [Representing FHE SIMD Operations](#representing-fhe-simd-operations) for
  an example) is an important aspect of accelerating SIMD-style operations in
  FHE, but it does not follow automatically from the batching rewrites applied
  so far. As a result, the `-rotate-and-reduce` pass needs to search for
  sequences of arithmetic operations that correspond to the full folding of a
  tensor, i.e., patterns such as `t[0]+(t[1]+(t[2]+t[3]+(...)))`, which
  currently uses a backwards search through the IR, but could be achieved more
  efficiently through a data flow analysis (c.f. issue
  [#532](https://github.com/google/heir/issues/523)). In HECO, rotate-and-reduce
  is handled differently, by identifying sequences of compatible operations
  prior to batching and rewriting them to "n-ary" operations. However, this
  approach requires non-standard arithmetic operations and is therefore not
  suitable for use in HEIR. However, there is likely still an opportunity to
  make the patterns in HEIR more robust/general (e.g., support constant scalar
  operands in the fold, or support non-full-tensor folds). See issue
  [#522](https://github.com/google/heir/issues/522) for ideas on how to make the
  HEIR pattern more robust/more general.

### Insert Rotate Pass

TODO(#721): Write a detailed description of the rotation insertion pass and the
associated target slot analysis.

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
