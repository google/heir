// RUN: heir-opt --fold-convert-layout-into-assign-layout %s | FileCheck %s

// CHECK: [[col_major_matrix:#[^ ]*]] = #tensor_ext.new_layout<"{ [row, col] -> [ct, slot] : ct = 0 and (-16col - row + slot) mod 1024 = 0 and 0 <= row <= 15 and 0 <= col <= 15 and 0 <= slot <= 1023 }">
#row_major_matrix = #tensor_ext.new_layout<"{ [row, col] -> [ct, slot] : ct = 0 and (-16row - col + slot) mod 1024 = 0 and 0 <= row <= 15 and 0 <= col <= 15 and 0 <= slot <= 1023 }">
#col_major_matrix = #tensor_ext.new_layout<"{ [row, col] -> [ct, slot] : ct = 0 and (-16col - row + slot) mod 1024 = 0 and 0 <= row <= 15 and 0 <= col <= 15 and 0 <= slot <= 1023 }">
#col_major_matrix2 = #tensor_ext.new_layout<"{ [row, col] -> [ct, slot] : ct = 0 and (-17col - row + slot) mod 1024 = 0 and 0 <= row <= 15 and 0 <= col <= 15 and 0 <= slot <= 1023 }">

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
