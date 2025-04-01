// RUN: heir-opt --fold-convert-layout-into-assign-layout %s | FileCheck %s

// CHECK: [[col_major_matrix:#[^ ]*]] = #tensor_ext.layout<map = (d0, d1) -> (d1 * 16 + d0)>
#row_major_matrix = #tensor_ext.layout<map = (d0, d1) -> (d0 * 16 + d1)>
#col_major_matrix = #tensor_ext.layout<map = (d0, d1) -> (d1 * 16 + d0)>
#col_major_matrix2 = #tensor_ext.layout<map = (d0, d1) -> (d1 * 17 + d0)>

// CHECK: @assign_layout
// CHECK-SAME: [[arg0:%[^:]*]]: tensor<16x16xi16>)
func.func @assign_layout(%arg0 : tensor<16x16xi16>) -> tensor<16x16xi16> {
  // CHECK: [[v0:[^ ]*]] = tensor_ext.assign_layout [[arg0]] {layout = [[col_major_matrix]], tensor_ext.layout = [[col_major_matrix]]} : tensor<16x16xi16>
  // CHECK-NEXT: return [[v0]]
  %0 = tensor_ext.assign_layout %arg0 {layout = #row_major_matrix} : tensor<16x16xi16>
  %1 = tensor_ext.convert_layout %0 {from_layout = #row_major_matrix, to_layout = #col_major_matrix} : tensor<16x16xi16>
  return %1 : tensor<16x16xi16>
}

// CHECK: @fold_multiple
// CHECK-SAME: [[arg0:%[^:]*]]: tensor<16x16xi16>)
func.func @fold_multiple(%arg0 : tensor<16x16xi16>) -> (tensor<16x16xi16>, tensor<16x16xi16>) {
  // CHECK-COUNT-2: tensor_ext.assign_layout
  %0 = tensor_ext.assign_layout %arg0 {layout = #row_major_matrix} : tensor<16x16xi16>
  %1 = tensor_ext.convert_layout %0 {from_layout = #row_major_matrix, to_layout = #col_major_matrix} : tensor<16x16xi16>
  %2 = tensor_ext.convert_layout %0 {from_layout = #row_major_matrix, to_layout = #col_major_matrix2} : tensor<16x16xi16>
  return %1, %2 : tensor<16x16xi16>, tensor<16x16xi16>
}
