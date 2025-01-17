// RUN: heir-opt --layout-propagation %s | FileCheck %s

!tensor = tensor<32x32xi16>
!tensor2 = tensor<32xi16>
!stensor = !secret.secret<!tensor>
!stensor2 = !secret.secret<!tensor2>

// Test that when an operation changes the tensor layour in an incompatible way,
// a layout conversion operation is inserted.

// CHECK: [[input_map:#[^ ]*]] = affine_map<(d0, d1) -> (d0 * 32 + d1)>
// CHECK: [[row_reduced_map:#[^ ]*]] = affine_map<(d0) -> (d0)>
// CHECK: [[col_reduced_map:#[^ ]*]] = affine_map<(d0) -> (d0 * 32)>

// CHECK: insert_conversion
// CHECK-SAME: %[[arg0:[^:]+]]: !secret.secret<tensor<32x32xi16>> {tensor_ext.layout = #tensor_ext.layout<layout = (d0, d1) -> (d0 * 32 + d1)>}
// CHECK-SAME: %[[arg1:[^:]+]]: !secret.secret<tensor<32x32xi16>> {tensor_ext.layout = #tensor_ext.layout<layout = (d0, d1) -> (d0 * 32 + d1)>}
func.func @insert_conversion(%arg0: !stensor, %arg1: !stensor) -> !stensor2 {
  // CHECK: secret.generic
  // CHECK-SAME: ins(%[[arg0]], %[[arg1]]
  // CHECK-SAME: [[arg0]] = {layout = [[input_map]]}
  // CHECK-SAME: [[arg1]] = {layout = [[input_map]]}
  // Note this one denotes the layout of the result of the generic op
  // CHECK-SAME: layout =
  // CHECK-SAME: [
  // CHECK-SAME: [[row_reduced_map]]
  // CHECK-SAME: ]
  %0 = secret.generic ins(%arg0, %arg1: !stensor, !stensor) {
  ^body(%pt_arg0: !tensor, %pt_arg1: !tensor):
    // result of sum has row-major layout, i.e., with don't cares at the end
    // (1, 2, ..., 32, x, x, x, x, ..., x)
    // CHECK: [[unconverted:[^ ]+]] = tensor_ext.sum
    // CHECK-SAME: {layout = [[[row_reduced_map]]]}
    %1 = tensor_ext.sum %pt_arg0, 0 : !tensor -> !tensor2

    // result of sum has column-major layout, i.e., strided
    // (1, x, ..., x, 2, x, ..., x, 3, x, ..., x, ...)
    // At this stage, layout inference would annotate this with #strided attr
    // CHECK: [[to_convert:%.+]] = tensor_ext.sum
    // CHECK-SAME: {layout = [[[col_reduced_map]]]}
    %2 = tensor_ext.sum %pt_arg1, 1 : !tensor -> !tensor2

    // CHECK: [[converted:%.+]] = tensor_ext.convert_layout [[to_convert]]
    // CHECK-SAME: from_layout = [[col_reduced_map]]
    // CHECK-SAME: layout = [[row_reduced_map]]
    // CHECK-SAME: to_layout = [[row_reduced_map]]
    // CHECK: arith.addi [[unconverted]], [[converted]]
    %3 = arith.addi %1, %2 : !tensor2
    secret.yield %3 : !tensor2
  } -> !stensor2
  return %0 : !stensor2
}
