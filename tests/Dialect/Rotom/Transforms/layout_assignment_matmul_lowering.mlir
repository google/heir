// RUN: heir-opt %s --rotom-assign-layout --rotom-materialize-tensor-ext-layout --convert-to-ciphertext-semantics="ciphertext-size=16" --mlir-print-local-scope | FileCheck %s

// Matvec, lowered end to end. Both operands are seeded sources, so the
// search packs them directly at the compute placement -- the matrix
// column-major ([k][i] slots) and the vector with i's positions as
// replication -- and no operand conversion is materialized at all. The
// kernel is one multiply and a log-tree rotate-and-reduce over k's slot
// period (rotate 8, add, rotate 4, add); the true sums sit at the k=0 slot
// offsets (the result layout leaves the other offsets as a gap). No kernel
// attribute is involved.

#seed_mat = #rotom.seed<layouts = [#rotom.layout<dims = [#rotom.dim<[0:4:1]>, #rotom.dim<[1:4:1]>], n = 16>]>
#seed_col = #rotom.seed<layouts = [#rotom.layout<dims = [#rotom.dim<[0:4:1]>], n = 16>]>

module {
  // CHECK: func.func @matvec_lowering
  // CHECK-SAME: tensor<1x16xf32>
  func.func @matvec_lowering(%a: tensor<4x4xf32> {rotom.seed = #seed_mat}, %b: tensor<4x1xf32> {rotom.seed = #seed_col}) -> tensor<4x1xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %empty = tensor.empty() : tensor<4x1xf32>
    %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<4x1xf32>) -> tensor<4x1xf32>
    // CHECK-NOT: tensor_ext.remap
    // CHECK: %[[PROD:.*]] = arith.mulf %arg0, %arg1
    // CHECK: %[[ROT2:.*]] = tensor_ext.rotate %[[PROD]]
    // CHECK: %[[SUM2:.*]] = arith.addf %[[PROD]], %[[ROT2]]
    // CHECK: %[[ROT1:.*]] = tensor_ext.rotate %[[SUM2]]
    // CHECK: %[[SUM1:.*]] = arith.addf %[[SUM2]], %[[ROT1]]
    // CHECK: return %[[SUM1]]
    %0 = linalg.matmul ins(%a, %b : tensor<4x4xf32>, tensor<4x1xf32>) outs(%fill : tensor<4x1xf32>) -> tensor<4x1xf32>
    return %0 : tensor<4x1xf32>
  }
}
