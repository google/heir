// RUN: heir-opt %s --rotom-assign-layout --rotom-materialize-tensor-ext-layout --convert-to-ciphertext-semantics="ciphertext-size=8" --mlir-print-local-scope | FileCheck %s

#layout_lhs = #rotom.layout<dims = [#rotom.dim<dim = 0, size = 2>, #rotom.dim<dim = 1, size = 4>], n = 8>
#layout_rhs = #rotom.layout<dims = [#rotom.dim<dim = 0, size = 4>, #rotom.dim<dim = 1, size = 2>], n = 8>
#seed_lhs = #rotom.seed<layouts = [#layout_lhs]>
#seed_rhs = #rotom.seed<layouts = [#layout_rhs]>

module {
  // CHECK: func.func @rectangular_rotom_matmul
  // CHECK-SAME: tensor<1x8xf32>
  // CHECK-NOT: linalg.matmul
  // CHECK: tensor_ext.remap
  // CHECK: arith.mulf
  // CHECK: arith.addf
  func.func @rectangular_rotom_matmul(%lhs: tensor<2x4xf32> {rotom.seed = #seed_lhs}, %rhs: tensor<4x2xf32> {rotom.seed = #seed_rhs}) -> tensor<2x2xf32> {
    %empty = tensor.empty() : tensor<2x2xf32>
    %0 = linalg.matmul ins(%lhs, %rhs : tensor<2x4xf32>, tensor<4x2xf32>) outs(%empty : tensor<2x2xf32>) -> tensor<2x2xf32>
    return %0 : tensor<2x2xf32>
  }
}

// -----

#layout_lhs_strided = #rotom.layout<dims = [#rotom.dim<dim = 0, size = 2, stride = 2>, #rotom.dim<dim = 1, size = 4>], n = 8>
#layout_rhs_strided = #rotom.layout<dims = [#rotom.dim<dim = 0, size = 4>, #rotom.dim<dim = 1, size = 2, stride = 2>], n = 8>
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
