// RUN: heir-opt %s --rotom-assign-layout --mlir-print-local-scope | FileCheck %s

#layout_lhs_row_major = #rotom.layout<dims = [#rotom.dim<dim = 0, size = 4>, #rotom.dim<dim = 1, size = 8>], n = 32>
#layout_rhs_row_major = #rotom.layout<dims = [#rotom.dim<dim = 0, size = 8>, #rotom.dim<dim = 1, size = 4>], n = 32>
#seed_lhs_row_major = #rotom.seed<layouts = [#layout_lhs_row_major]>
#seed_rhs_row_major = #rotom.seed<layouts = [#layout_rhs_row_major]>

module {
  // CHECK: func.func @matmul_non_bicyclic
  func.func @matmul_non_bicyclic(%lhs: tensor<4x8xf32> {rotom.seed = #seed_lhs_row_major}, %rhs: tensor<8x4xf32> {rotom.seed = #seed_rhs_row_major}) -> tensor<4x4xf32> {
    %empty = tensor.empty() : tensor<4x4xf32>
    // CHECK: %[[MM:.*]] = linalg.matmul {rotom.layout =
    // CHECK-SAME: secret.kernel = #secret.kernel<name = "RotomMatmul"
    // CHECK: return %[[MM]]
    %0 = linalg.matmul ins(%lhs, %rhs : tensor<4x8xf32>, tensor<8x4xf32>) outs(%empty : tensor<4x4xf32>) -> tensor<4x4xf32>
    return %0 : tensor<4x4xf32>
  }
}

// -----

#layout_one = #rotom.layout<dims = [#rotom.dim<dim = 0, size = 1>, #rotom.dim<dim = 1, size = 1>], n = 1>
#seed_one = #rotom.seed<layouts = [#layout_one]>

module {
  // CHECK: func.func @matmul_bicyclic_degenerate
  func.func @matmul_bicyclic_degenerate(%lhs: tensor<1x1xf32> {rotom.seed = #seed_one}, %rhs: tensor<1x1xf32> {rotom.seed = #seed_one}) -> tensor<1x1xf32> {
    %empty = tensor.empty() : tensor<1x1xf32>
    // CHECK: linalg.matmul
    // CHECK-SAME: rotom.layout =
    // CHECK-SAME: secret.kernel = #secret.kernel<name = "MatmulBicyclic"
    %0 = linalg.matmul ins(%lhs, %rhs : tensor<1x1xf32>, tensor<1x1xf32>) outs(%empty : tensor<1x1xf32>) -> tensor<1x1xf32>
    return %0 : tensor<1x1xf32>
  }
}
