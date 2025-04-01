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

def AlignTensorSizes : Pass<"align-tensor-sizes"> {
  let summary = "Resize tensors into tensors with a fixed size final dimension";

  let description = [{ This pass resizes input tensors with arbitrary sizes into
  tensors with whose final dimensions has a fixed size. All input tensors are
  required to be one-dimensional. The `--size` option specifies the size of the
  final dimension of the output tensors, and is required to be a power of two.

  To align the tensors in the input IR, the pass first zero pads the input to
  the nearest power of two before replicating or splitting it into the output
  shape determined by `size`. The resulting transformation is described in a
  `SIMDPackingAttr` encoding attribute on the final tensor.

  For example, with `size=16`,a tensor with 7 elements will be zero-padded to 8
  elements, and then replicated twice to fill a tensor with size 16. The
  `SIMDPackingAttr` will encode the input shape, the number of elements that
  were zero-padded, and the output shape.

  Input:
  ```mlir
  %0 = tensor.empty() : tensor<7xi32>
  ```

  Output:
  ```mlir
  %0 = tensor.empty() : tensor<16xi32, #tensor_ext.simd_packing<in = [7], padding = [1], out = [16]>>
  ```

  A tensor with 30 elements will be zero padded with 2 elements and split into
  two tensors of size 16.

  Input:
  ```mlir
  %0 = tensor.empty() : tensor<30xi32>
  ```

  Output:
  ```mlir
  %0 = tensor.empty() : tensor<2x16xi32, #tensor_ext.simd_packing<in = [30], padding = [2], out = [16]>>
  ```

  Note that this pass does not insert any new operations like `reshape`, but
  rather transforms the IR to use tensors with a fixed dimension. This pass may
  be used to align the sizes of tensors that represent plaintexts and
  ciphertexts in RLWE schemes that support SIMD slots and operations.
  }];

  let dependentDialects = ["mlir::heir::tensor_ext::TensorExtDialect"];

  let options = [
    Option<"size", "size", "int",
           /*default=*/"1024", "Power of two size of the final dimension of the output tensors.">
  ];
}

def ImplementShiftNetwork : Pass<"implement-shift-network"> {
  let summary = "Implement tensor_ext.convert_layout ops as shift newtorks";

  let description = [{
  This pass converts `tensor_ext.permute` ops into a network of
  `tensor_ext.rotate` ops, aiming to minimize the overall latency of the
  permutation.

  The input IR must have tensors that correspond to plaintexts or
  ciphertexts.

  The method uses graph coloring, an approach based on Vos-Vos-Erkin 2022,
  ["Efficient Circuits for Permuting and Mapping Packed Values Across
   Leveled Homomorphic Ciphertexts"](https://link.springer.com/chapter/10.1007/978-3-031-17140-6_20).

  Example, Figure 3 from the paper above:

  ```mlir
  // Provide an explicit permutation, though an affine_map can also be used.
  #map = dense<[13, 8, 4, 0, 11, 7, 14, 5, 15, 3, 12, 6, 10, 2, 9, 1]> : tensor<16xi64>
  func.func @figure3(%0: tensor<16xi32>) -> tensor<16xi32> {
    %1 = tensor_ext.permute %0 {permutation = #map} : tensor<16xi32>
    return %1 : tensor<16xi32>
  }
  ```

  Then running `--implement-shift-network=ciphertext-size=16` produces
  a shift network composed of plaintext-ciphertext masks (`arith.constant` + `arith.muli`)
  followed by rotations and additions. The Vos-Vos-Erkin method splits the work
  into multiple independent groups that are added together at the end.

  ```mlir
  func.func @figure3(%arg0: tensor<16xi32>) -> tensor<16xi32> {
    %cst = arith.constant dense<[1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0]> : tensor<16xi32>
    %0 = arith.muli %arg0, %cst : tensor<16xi32>
    %c1_i32 = arith.constant 1 : i32
    %1 = tensor_ext.rotate %0, %c1_i32 : tensor<16xi32>, i32
    %cst_0 = arith.constant dense<[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]> : tensor<16xi32>
    %2 = arith.muli %arg0, %cst_0 : tensor<16xi32>
    %c2_i32 = arith.constant 2 : i32
    %3 = tensor_ext.rotate %2, %c2_i32 : tensor<16xi32>, i32
    %4 = arith.addi %1, %3 : tensor<16xi32>
    %cst_1 = arith.constant dense<[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]> : tensor<16xi32>
    %5 = arith.muli %arg0, %cst_1 : tensor<16xi32>
    %c4_i32 = arith.constant 4 : i32
    %6 = tensor_ext.rotate %5, %c4_i32 : tensor<16xi32>, i32
    %7 = arith.addi %4, %6 : tensor<16xi32>
    %cst_2 = arith.constant dense<[0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]> : tensor<16xi32>
    %8 = arith.muli %arg0, %cst_2 : tensor<16xi32>
    %c8_i32 = arith.constant 8 : i32
    %9 = tensor_ext.rotate %8, %c8_i32 : tensor<16xi32>, i32
    %10 = arith.addi %7, %9 : tensor<16xi32>
    %cst_3 = arith.constant dense<[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]> : tensor<16xi32>
    %11 = arith.muli %arg0, %cst_3 : tensor<16xi32>
    %c1_i32_4 = arith.constant 1 : i32
    %12 = tensor_ext.rotate %11, %c1_i32_4 : tensor<16xi32>, i32
    %cst_5 = arith.constant dense<[0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0]> : tensor<16xi32>
    %13 = arith.muli %arg0, %cst_5 : tensor<16xi32>
    %c2_i32_6 = arith.constant 2 : i32
    %14 = tensor_ext.rotate %13, %c2_i32_6 : tensor<16xi32>, i32
    %15 = arith.addi %12, %14 : tensor<16xi32>
    %cst_7 = arith.constant dense<[0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0]> : tensor<16xi32>
    %16 = arith.muli %arg0, %cst_7 : tensor<16xi32>
    %c4_i32_8 = arith.constant 4 : i32
    %17 = tensor_ext.rotate %16, %c4_i32_8 : tensor<16xi32>, i32
    %18 = arith.addi %15, %17 : tensor<16xi32>
    %cst_9 = arith.constant dense<[0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0]> : tensor<16xi32>
    %19 = arith.muli %arg0, %cst_9 : tensor<16xi32>
    %c8_i32_10 = arith.constant 8 : i32
    %20 = tensor_ext.rotate %19, %c8_i32_10 : tensor<16xi32>, i32
    %21 = arith.addi %18, %20 : tensor<16xi32>
    %22 = arith.addi %10, %21 : tensor<16xi32>
    %cst_11 = arith.constant dense<[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]> : tensor<16xi32>
    %23 = arith.muli %arg0, %cst_11 : tensor<16xi32>
    %c1_i32_12 = arith.constant 1 : i32
    %24 = tensor_ext.rotate %23, %c1_i32_12 : tensor<16xi32>, i32
    %cst_13 = arith.constant dense<[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1]> : tensor<16xi32>
    %25 = arith.muli %arg0, %cst_13 : tensor<16xi32>
    %c2_i32_14 = arith.constant 2 : i32
    %26 = tensor_ext.rotate %25, %c2_i32_14 : tensor<16xi32>, i32
    %27 = arith.addi %24, %26 : tensor<16xi32>
    %cst_15 = arith.constant dense<[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]> : tensor<16xi32>
    %28 = arith.muli %arg0, %cst_15 : tensor<16xi32>
    %c4_i32_16 = arith.constant 4 : i32
    %29 = tensor_ext.rotate %28, %c4_i32_16 : tensor<16xi32>, i32
    %30 = arith.addi %27, %29 : tensor<16xi32>
    %cst_17 = arith.constant dense<[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1]> : tensor<16xi32>
    %31 = arith.muli %arg0, %cst_17 : tensor<16xi32>
    %c8_i32_18 = arith.constant 8 : i32
    %32 = tensor_ext.rotate %31, %c8_i32_18 : tensor<16xi32>, i32
    %33 = arith.addi %30, %32 : tensor<16xi32>
    %34 = arith.addi %22, %33 : tensor<16xi32>
    return %34 : tensor<16xi32>
  }
  ```
  }];

  let dependentDialects = ["mlir::heir::tensor_ext::TensorExtDialect"];

  // TODO(#4102): reevaluate flag name
  let options = [
    Option<
      "ciphertextSize",
      "ciphertext-size",
      "int",
      /*default=*/"1024",
      "Power of two length of the ciphertexts the data is packed in."
    >
  ];
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

#endif  // LIB_DIALECT_TENSOREXT_TRANSFORMS_PASSES_TD_
