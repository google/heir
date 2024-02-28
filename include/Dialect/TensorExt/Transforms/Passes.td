#ifndef INCLUDE_DIALECT_TENSOREXT_TRANSFORMS_PASSES_TD_
#define INCLUDE_DIALECT_TENSOREXT_TRANSFORMS_PASSES_TD_

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

#endif  // INCLUDE_DIALECT_TENSOREXT_TRANSFORMS_PASSES_TD_
