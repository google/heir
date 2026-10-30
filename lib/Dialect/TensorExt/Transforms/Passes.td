#ifndef LIB_DIALECT_TENSOREXT_TRANSFORMS_PASSES_TD_
#define LIB_DIALECT_TENSOREXT_TRANSFORMS_PASSES_TD_

include "mlir/Pass/PassBase.td"

def SimdVectorizeLoops : Pass<"simd-vectorize-loops"> {
  let summary = "Vectorize loops over tensor elements into rotations, without unrolling";
  let description = [{
  This pass converts loop nests that compute tensor elements one at a time into
  SIMD operations on whole tensors and cyclic rotations. It is based on the
  batching optimization of the
  [HECO paper](https://arxiv.org/abs/2202.01649), but analyzes the loops as
  affine iteration domains instead of requiring them to be fully unrolled
  first.

  The pass maps the iterations of some loops of a nest (the vectorized loops)
  injectively to the slots of a tensor through an affine function of their
  induction variables, the *slot map*. For a *map* nest, a perfect nest of
  loops carrying a single tensor whose innermost body inserts one scalar into
  it, the slot map is first taken to be the insertion index. Otherwise, or if
  that fails, the slot map is the index of a `tensor.extract` in the nest
  (preferring secret tensors), and the vectorized loops are the loops it
  depends on, together with the loops carrying the output of a map nest. The
  pass checks with an integer set analysis of the iteration domain that the
  slot map is injective and lands within the tensor. Other loops in the nest
  are kept as loops over tensors, and all loops except the ones carrying the
  output must reduce (with `arith.addi`, `arith.addf`, `arith.muli` or
  `arith.mulf`) into a single scalar.

  Each `tensor.extract` whose index is an affine function of the induction
  variables with a constant offset from the slot map becomes a rotation of the
  source tensor by that offset:

  ```mlir
  %0 = affine.for %i = 0 to 16 iter_args(%t = %init) -> tensor<16xi16> {
    %j = affine.apply affine_map<(d0) -> ((d0 + 1) mod 16)>(%i)
    %a = tensor.extract %x[%j] : tensor<16xi16>
    %b = arith.muli %a, %a : i16
    %u = tensor.insert %b into %t[%i] : tensor<16xi16>
    affine.yield %u : tensor<16xi16>
  }
  ```

  becomes

  ```mlir
  %c1 = arith.constant 1 : index
  %r = tensor_ext.rotate %x, %c1 : tensor<16xi16>, index
  %0 = arith.muli %r, %r : tensor<16xi16>
  ```

  An extraction from a plaintext tensor whose offset is not constant instead
  becomes a plaintext vector that gathers each iteration's element into its
  slot. An extraction from a secret tensor whose offset is not constant (or
  that has a different number of elements) is relaid out with one rotation and
  mask for each distinct offset. Arithmetic on the extracted values is applied
  to the resulting tensors.

  A vectorized loop that reduces into a scalar becomes a
  `tensor_ext.rotate_and_reduce` over the slots of its iterations, whose period
  is the stride of the slot map in that loop, leaving the result in the slot of
  the first iteration. A kept loop becomes a loop over tensors, and is replaced
  by a `tensor_ext.rotate_and_reduce` when it sums rotations of a
  loop-invariant tensor by an affine function of its induction variable,
  optionally weighted by plaintexts.

  Finally, the result of a map nest is moved to the slots given by its
  insertion index, if it was computed with a different slot map. If the
  insertion index does not cover every slot, a plaintext mask preserves the
  original values of the other slots.

  Loop nests that do not fit these patterns are left unchanged.
  }];
  let dependentDialects = [
    "mlir::heir::tensor_ext::TensorExtDialect",
    "mlir::arith::ArithDialect",
    "mlir::tensor::TensorDialect",
  ];
}

def ImplementShiftNetwork : Pass<"implement-shift-network"> {
  let summary = "Implement tensor_ext.convert_layout ops as shift newtorks";

  let description = [{
  This pass converts `tensor_ext.remap` ops into a network of
  `tensor_ext.rotate` ops, aiming to minimize the overall latency of the
  permutation. The input IR must have tensors that correspond to plaintexts or
  ciphertexts, and be in "ciphertext semantics," i.e., after an IR has been
  processed via `convert-to-ciphertext-semantics`.

  The method uses graph coloring, an approach based on Vos-Vos-Erkin 2022,
  ["Efficient Circuits for Permuting and Mapping Packed Values Across
   Leveled Homomorphic Ciphertexts"](https://link.springer.com/chapter/10.1007/978-3-031-17140-6_20).

  (* example filepath=tests/Dialect/TensorExt/Transforms/implement_shift_network_example.mlir *)
  }];

  let dependentDialects = ["mlir::heir::tensor_ext::TensorExtDialect"];
}

def FoldConvertLayoutIntoAssignLayout : Pass<"fold-convert-layout-into-assign-layout"> {
  let summary = "Merges tensor_ext.convert_layout ops into preceding tensor_ext.assign_layout ops";
  let description = [{
  A `tensor_ext.assign_layout` op corresponds to an encoding of a cleartext
  into a plaintext or ciphertext. If this is immediately followed by a
  `tensor_ext.convert_layout` op, then one can just change the initial encoding
  to correspond to the result of the conversion.

  If the result of an `assign_layout` has multiple subsequent `convert_layout`
  ops, then they are folded into multiple `assign_layout` ops applied to the
  same cleartext.
  }];
  let dependentDialects = ["mlir::heir::tensor_ext::TensorExtDialect"];
}

def ImplementRotateAndReduce : Pass<"implement-rotate-and-reduce"> {
  let summary = "Implement tensor_ext.rotate_and_reduce ops with baby-steps / giant-steps";

  let description = [{
  This pass converts `tensor_ext.rotate_and_reduce` ops
  into a sequence of arithmetic operations and `tensor_ext.rotate` ops, aiming
  to minimize the number of ciphertext rotation operations using a Baby-Steps /
  Giant-Steps approach.

  A `tensor_ext.rotate_and_reduce` op computes the reduction of rotated tensors
  in the form `\sum_{i = 0}^{n} P(i) \cdot rotate(v, T*i)` where `T` is some
  period of rotation. The naive approach would compute `n` rotations of the
  ciphertext `v`. The Baby-Steps / Giant-Steps approach from [Faster Homomorphic
  Linear Transformations in HElib](https://eprint.iacr.org/2018/244.pdf) can
  compute the sum with `O(sqrt(n))` ciphertext rotations instead by evaluating
  the sum with the following expression

  $$
    \sum_{i = 0}^{n} P(i) \cdot rotate(v, T*i) =
    \sum_{j = 0}^{\sqrt{n}} \sum_{k = 0}^{\sqrt{n}} P(j + gk) rotate(v, T*(j + gk)) =
    \sum_{j = 0}^{\sqrt{n}} rotate( \sum_{k = 0}^{\sqrt{n}} rotate(P(j + gk), -gk) rotate(v, T*j) , -gk)
  $$

  This approach uses `\sqrt{n}` ciphertext rotations of `v` in the inner sum
  (the baby steps) and then `\sqrt{n}` ciphertext rotations of the partial sums
  (the giant steps) to compute the full sum.

  The input IR must have tensors that correspond to plaintexts or
  ciphertexts.

  (* example filepath=tests/Dialect/TensorExt/Transforms/implement_rotate_and_reduce.mlir *)
  }];

  let options = [
    Option<"unroll", "unroll", "bool", /*default=*/"true",
           "Unroll the loop instead of generating a rolled loop">
  ];

  let dependentDialects = [
    "mlir::heir::tensor_ext::TensorExtDialect",
    "mlir::heir::kernel::KernelDialect",
    "mlir::scf::SCFDialect",
    "mlir::arith::ArithDialect",
    "mlir::tensor::TensorDialect"
  ];
}

#endif  // LIB_DIALECT_TENSOREXT_TRANSFORMS_PASSES_TD_
