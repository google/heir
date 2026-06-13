// RUN: heir-opt %s --rotom-assign-layout --rotom-materialize-tensor-ext-layout --convert-to-ciphertext-semantics="ciphertext-size=8" --mlir-print-local-scope | FileCheck %s

#layout_lhs = #rotom.layout<dims = [#rotom.dim<[0:2:1]>, #rotom.dim<[1:4:1]>], n = 8>
#layout_rhs = #rotom.layout<dims = [#rotom.dim<[0:4:1]>, #rotom.dim<[1:2:1]>], n = 8>
#seed_lhs = #rotom.seed<layouts = [#layout_lhs]>
#seed_rhs = #rotom.seed<layouts = [#layout_rhs]>

module {
  // The Rotom matmul lowers via the rotate-multiply-accumulate kernel: each
  // contraction term is a pair of ciphertext rotations, multiplied and masked
  // into the running accumulator. No per-scalar remaps remain.
  // CHECK: func.func @rectangular_rotom_matmul
  // CHECK-SAME: tensor<1x8xf32>
  // CHECK-NOT: linalg.matmul
  // CHECK-NOT: tensor_ext.remap
  // CHECK: tensor_ext.rotate
  // CHECK: tensor_ext.rotate
  // CHECK: arith.mulf
  // CHECK: arith.addf
  func.func @rectangular_rotom_matmul(%lhs: tensor<2x4xf32> {rotom.seed = #seed_lhs}, %rhs: tensor<4x2xf32> {rotom.seed = #seed_rhs}) -> tensor<2x2xf32> {
    %empty = tensor.empty() : tensor<2x2xf32>
    %0 = linalg.matmul ins(%lhs, %rhs : tensor<2x4xf32>, tensor<4x2xf32>) outs(%empty : tensor<2x2xf32>) -> tensor<2x2xf32>
    return %0 : tensor<2x2xf32>
  }
}

// -----

#layout_2x2 = #rotom.layout<dims = [#rotom.dim<[0:2:1]>, #rotom.dim<[1:2:1]>], n = 4>
#seed_2x2 = #rotom.seed<layouts = [#layout_2x2]>

module {
  // The rotate-multiply-accumulate kernel seeds the accumulator with the init
  // (destination) operand %arg2 and adds each masked rotated product into it.
  // CHECK: func.func @rotom_matmul_accumulates_init
  // CHECK-SAME: tensor<1x8xf32>
  // CHECK-NOT: linalg.matmul
  // CHECK-NOT: tensor_ext.remap
  // CHECK: %[[LHS_ROT:.*]] = tensor_ext.rotate %arg0
  // CHECK: %[[RHS_ROT:.*]] = tensor_ext.rotate %arg1
  // CHECK: %[[PRODUCT:.*]] = arith.mulf %[[LHS_ROT]], %[[RHS_ROT]]
  // CHECK: %[[MASKED:.*]] = arith.mulf %[[PRODUCT]]
  // CHECK: arith.addf %arg2, %[[MASKED]]
  func.func @rotom_matmul_accumulates_init(%lhs: tensor<2x2xf32> {rotom.seed = #seed_2x2}, %rhs: tensor<2x2xf32> {rotom.seed = #seed_2x2}, %init: tensor<2x2xf32>) -> tensor<2x2xf32> {
    %0 = linalg.matmul ins(%lhs, %rhs : tensor<2x2xf32>, tensor<2x2xf32>) outs(%init : tensor<2x2xf32>) -> tensor<2x2xf32>
    return %0 : tensor<2x2xf32>
  }
}

// -----

#layout_lhs_strided = #rotom.layout<dims = [#rotom.dim<[0:2:2]>, #rotom.dim<[1:4:1]>], n = 8>
#layout_rhs_strided = #rotom.layout<dims = [#rotom.dim<[0:4:1]>, #rotom.dim<[1:2:2]>], n = 8>
#seed_lhs_strided = #rotom.seed<layouts = [#layout_lhs_strided]>
#seed_rhs_strided = #rotom.seed<layouts = [#layout_rhs_strided]>

module {
  // CHECK: func.func @strided_matmul_not_rotom_lowered
  // CHECK-SAME: tensor<2x4xf32>
  // CHECK-NOT: tensor_ext.remap
  // CHECK: linalg.matmul
  func.func @strided_matmul_not_rotom_lowered(%lhs: tensor<2x4xf32> {rotom.seed = #seed_lhs_strided}, %rhs: tensor<4x2xf32> {rotom.seed = #seed_rhs_strided}) -> tensor<2x2xf32> {
    %empty = tensor.empty() : tensor<2x2xf32>
    %0 = linalg.matmul ins(%lhs, %rhs : tensor<2x4xf32>, tensor<4x2xf32>) outs(%empty : tensor<2x2xf32>) -> tensor<2x2xf32>
    return %0 : tensor<2x2xf32>
  }
}
