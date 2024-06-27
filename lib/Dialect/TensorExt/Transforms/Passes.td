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

#endif  // LIB_DIALECT_TENSOREXT_TRANSFORMS_PASSES_TD_
