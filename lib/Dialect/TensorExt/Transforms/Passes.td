#ifndef LIB_DIALECT_TENSOREXT_TRANSFORMS_PASSES_TD_
#define LIB_DIALECT_TENSOREXT_TRANSFORMS_PASSES_TD_

include "mlir/Pass/PassBase.td"

def InsertRotate : Pass<"insert-rotate"> {
  let summary = "Vectorize arithmetic FHE operations using HECO-style heuristics";
  let description = [{
  This pass implements the SIMD-vectorization passes from the
  [HECO paper](https://arxiv.org/abs/2202.01649).

  The pass operates by identifying arithmetic operations that can be suitably
  combined into a combination of cyclic rotations and vectorized operations
  on tensors. It further identifies a suitable "slot target" for each operation
  and heuristically aligns the operations to reduce unnecessary rotations.

  This pass by itself does not eliminate any operations, but instead inserts
  well-chosen rotations so that, for well-structured code (like unrolled affine loops),
  a subsequent `--cse` and `--canonicalize` pass will dramatically reduce the IR.
  As such, the pass is designed to be paired with the canonicalization patterns
  in `tensor_ext`, as well as the `collapse-insertion-chains` pass, which
  cleans up remaining insertion and extraction ops after the main simplifications
  are applied.

  Unlike HECO, this pass operates on plaintext types and tensors, along with
  the HEIR-specific `tensor_ext` dialect for its cyclic `rotate` op. It is intended
  to be run before lowering to a scheme dialect like `bgv`.
  }];
  let dependentDialects = ["mlir::heir::tensor_ext::TensorExtDialect"];
}

// TODO(#512): Investigate replacing this pattern with a tensor_ext.combine op
def CollapseInsertionChains : Pass<"collapse-insertion-chains"> {
  let summary = "Collapse chains of extract/insert ops into rotate ops when possible";
  let description = [{
  This pass is a cleanup pass for `insert-rotate`. That pass sometimes leaves
  behind a chain of insertion operations like this:

  ```mlir
  %extracted = tensor.extract %14[%c5] : tensor<16xi16>
  %inserted = tensor.insert %extracted into %dest[%c0] : tensor<16xi16>
  %extracted_0 = tensor.extract %14[%c6] : tensor<16xi16>
  %inserted_1 = tensor.insert %extracted_0 into %inserted[%c1] : tensor<16xi16>
  %extracted_2 = tensor.extract %14[%c7] : tensor<16xi16>
  %inserted_3 = tensor.insert %extracted_2 into %inserted_1[%c2] : tensor<16xi16>
  ...
  %extracted_28 = tensor.extract %14[%c4] : tensor<16xi16>
  %inserted_29 = tensor.insert %extracted_28 into %inserted_27[%c15] : tensor<16xi16>
  yield %inserted_29 : tensor<16xi16>
  ```

  In many cases, this chain will insert into every index of the `dest` tensor,
  and the extracted values all come from consistently aligned indices of the same
  source tensor. In this case, the chain can be collapsed into a single `rotate`.

  Each index used for insertion or extraction must be constant; this may
  require running `--canonicalize` or `--sccp` before this pass to apply
  folding rules (use `--sccp` if you need to fold constant through control flow).
  }];
  let dependentDialects = ["mlir::heir::tensor_ext::TensorExtDialect"];
}

def RotateAndReduce : Pass<"rotate-and-reduce"> {
  let summary = "Use a logarithmic number of rotations to reduce a tensor.";
  let description = [{
  This pass identifies when a commutative, associative binary operation is used
  to reduce all of the entries of a tensor to a single value, and optimizes the
  operations by using a logarithmic number of reduction operations.

  In particular, this pass identifies an unrolled set of operations of the form
  (the binary ops may come in any order):

  ```mlir
  %0 = tensor.extract %t[0] : tensor<8xi32>
  %1 = tensor.extract %t[1] : tensor<8xi32>
  %2 = tensor.extract %t[2] : tensor<8xi32>
  %3 = tensor.extract %t[3] : tensor<8xi32>
  %4 = tensor.extract %t[4] : tensor<8xi32>
  %5 = tensor.extract %t[5] : tensor<8xi32>
  %6 = tensor.extract %t[6] : tensor<8xi32>
  %7 = tensor.extract %t[7] : tensor<8xi32>
  %8 = arith.addi %0, %1 : i32
  %9 = arith.addi %8, %2 : i32
  %10 = arith.addi %9, %3 : i32
  %11 = arith.addi %10, %4 : i32
  %12 = arith.addi %11, %5 : i32
  %13 = arith.addi %12, %6 : i32
  %14 = arith.addi %13, %7 : i32
  ```

  and replaces it with a logarithmic number of `rotate` and `addi` operations:

  ```mlir
  %0 = tensor_ext.rotate %t, 4 : tensor<8xi32>
  %1 = arith.addi %t, %0 : tensor<8xi32>
  %2 = tensor_ext.rotate %1, 2 : tensor<8xi32>
  %3 = arith.addi %1, %2 : tensor<8xi32>
  %4 = tensor_ext.rotate %3, 1 : tensor<8xi32>
  %5 = arith.addi %3, %4 : tensor<8xi32>
  ```
  }];
  let dependentDialects = ["mlir::heir::tensor_ext::TensorExtDialect"];
}

def ImplementShiftNetwork : Pass<"implement-shift-network"> {
  let summary = "Implement tensor_ext.convert_layout ops as shift newtorks";

  let description = [{
  This pass converts `tensor_ext.permute` ops into a network of
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

  \[
    \sum_{i = 0}^{n} P(i) \cdot rotate(v, T*i) =
    \sum_{j = 0}^{\sqrt{n}} \sum_{k = 0}^{\sqrt{n}} P(j + gk) rotate(v, T*(j + gk)) =
    \sum_{j = 0}^{\sqrt{n}} rotate( \sum_{k = 0}^{\sqrt{n}} rotate(P(j + gk), -gk) rotate(v, T*j) , -gk)
  \]

  This approach uses `\sqrt{n}` ciphertext rotations of `v` in the inner sum
  (the baby steps) and then `\sqrt{n}` ciphertext rotations of the partial sums
  (the giant steps) to compute the full sum.

  The input IR must have tensors that correspond to plaintexts or
  ciphertexts.

  (* example filepath=tests/Dialect/TensorExt/Transforms/implement_rotate_and_reduce.mlir *)
  }];

  let dependentDialects = ["mlir::heir::tensor_ext::TensorExtDialect"];
}

#endif  // LIB_DIALECT_TENSOREXT_TRANSFORMS_PASSES_TD_
