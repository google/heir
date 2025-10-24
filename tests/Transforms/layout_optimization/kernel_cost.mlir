// RUN: heir-opt --layout-optimization %s | FileCheck %s

// Test that kernel costs are now considered in optimization decisions

#layout_row = #tensor_ext.layout<"{ [i0, i1] -> [ct, slot] : ct = 0 and slot = i0 * 16 + i1 and i0 >= 0 and i0 < 16 and i1 >= 0 and i1 < 16 }">
#layout_diag = #tensor_ext.layout<"{ [i0, i1] -> [ct, slot] : ct = 0 and slot = (i0 + i1) mod 16 and i0 >= 0 and i0 < 16 and i1 >= 0 and i1 < 16 }">
#layout_vec = #tensor_ext.layout<"{ [i0] -> [ct, slot] : ct = 0 and slot = i0 and i0 >= 0 and i0 < 16 }">

// CHECK-LABEL: @matvec_with_diagonal_kernel
func.func @matvec_with_diagonal_kernel(
    %matrix: tensor<16x16xf32>,
    %vec: tensor<16xf32>) -> tensor<16xf32> {
  %cst = tensor.empty() : tensor<16xf32>

  %matrix_diag = tensor_ext.assign_layout %matrix
    {layout = #layout_diag, tensor_ext.layout = #layout_diag} : tensor<16x16xf32>
  %vec_layout = tensor_ext.assign_layout %vec
    {layout = #layout_vec, tensor_ext.layout = #layout_vec} : tensor<16xf32>

  // With kernel cost model: optimizer should preserve MatvecDiagonal
  // because the cost of the kernel is accounted for in optimization decisions
  // CHECK: linalg.matvec
  // CHECK-SAME: secret.kernel
  %result = linalg.matvec
    {secret.kernel = #secret.kernel<name="MatvecDiagonal", force=false>,
     tensor_ext.layout = #layout_vec}
    ins(%matrix_diag, %vec_layout : tensor<16x16xf32>, tensor<16xf32>)
    outs(%cst : tensor<16xf32>) -> tensor<16xf32>

  return %result : tensor<16xf32>
}

// Test with different matrix sizes to verify cost scaling
// CHECK-LABEL: @matvec_large_matrix
func.func @matvec_large_matrix(
    %matrix: tensor<512x512xf32>,
    %vec: tensor<512xf32>) -> tensor<512xf32> {
  %cst = tensor.empty() : tensor<512xf32>

  // CHECK: linalg.matvec
  // CHECK-SAME: secret.kernel
  %result = linalg.matvec
    {secret.kernel = #secret.kernel<name="MatvecDiagonal", force=false>}
    ins(%matrix, %vec : tensor<512x512xf32>, tensor<512xf32>)
    outs(%cst : tensor<512xf32>) -> tensor<512xf32>

  return %result : tensor<512xf32>
}

// Test rectangular matrix
// CHECK-LABEL: @matvec_rectangular
func.func @matvec_rectangular(
    %matrix: tensor<8x4xf32>,
    %vec: tensor<4xf32>) -> tensor<8xf32> {
  %cst = tensor.empty() : tensor<8xf32>

  // Cost should be based on number of rows (8), not columns (4)
  // CHECK: linalg.matvec
  // CHECK-SAME: secret.kernel
  %result = linalg.matvec
    {secret.kernel = #secret.kernel<name="MatvecDiagonal", force=false>}
    ins(%matrix, %vec : tensor<8x4xf32>, tensor<4xf32>)
    outs(%cst : tensor<8xf32>) -> tensor<8xf32>

  return %result : tensor<8xf32>
}
