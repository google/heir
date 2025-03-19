// RUN: heir-opt --layout-propagation %s | FileCheck %s

!tensor = tensor<32x32xi16>
!tensor2 = tensor<32xi16>
!stensor = !secret.secret<!tensor>
!stensor2 = !secret.secret<!tensor2>

// Test that when an operation changes the tensor layour in an incompatible way,
// a layout conversion operation is inserted.

// CHECK: insert_conversion
// CHECK-SAME: %[[arg0:[^:]+]]: !secret.secret<tensor<32x32xi16>> {tensor_ext.layout = #tensor_ext.layout<map = (d0, d1) -> (d0 * 32 + d1)>}
// CHECK-SAME: %[[arg1:[^:]+]]: !secret.secret<tensor<32x32xi16>> {tensor_ext.layout = #tensor_ext.layout<map = (d0, d1) -> (d0 * 32 + d1)>}
func.func @insert_conversion(%arg0: !stensor, %arg1: !stensor) -> !stensor2 {
  // CHECK: [[init0:%.*]] = arith.constant dense<0>
  // CHECK: [[init1:%.*]] = arith.constant dense<0>
  %out_1 = arith.constant dense<0> : !tensor2
  %out_2 = arith.constant dense<0> : !tensor2

  // CHECK: secret.generic
  // CHECK-SAME: ins(%[[arg0]], %[[arg1]]
  // CHECK-SAME: attrs = {__argattrs = [
  // CHECK-SAME {tensor_ext.layout = #tensor_ext.layout<map = [[input_map]]}>,
  // CHECK-SAME {tensor_ext.layout = #tensor_ext.layout<map = [[input_map]]}>],
  // Note this one denotes the layout of the result of the generic op
  // CHECK-SAME: tensor_ext.layout = [#tensor_ext.layout<map = (d0) -> (d0)>]
  %0 = secret.generic ins(%arg0, %arg1: !stensor, !stensor) {
  ^body(%pt_arg0: !tensor, %pt_arg1: !tensor):
    // CHECK: tensor_ext.assign_layout [[init0]] {layout = #tensor_ext.layout<map = (d0) -> (d0)>, tensor_ext.layout = #tensor_ext.layout<map = (d0) -> (d0)>}

    // result of sum has row-major layout, i.e., with implicit repetition at the end
    // (1, 2, ..., 32, 1, 2, ..., 32, ...)
    // CHECK: [[unconverted:[^ ]+]] = linalg.reduce
    // CHECK-SAME: {tensor_ext.layout = [#tensor_ext.layout<map = (d0) -> (d0)>]
    %1 = linalg.reduce { arith.addi } ins(%pt_arg0:!tensor) outs(%out_1:!tensor2) dimensions = [0]

    // CHECK: tensor_ext.assign_layout [[init1]]
    // CHECK-sAME: layout = [[row_reduced_map]]
    // CHECK-sAME: tensor_ext.layout = [[row_reduced_map]]
    // CHECK: tensor_ext.convert_layout
    // CHECK-SAME: from_layout = #tensor_ext.layout<map = (d0) -> (d0)>
    // CHECK-SAME: to_layout = #tensor_ext.layout<map = (d0) -> (d0 * 32)>

    // result of sum has column-major layout, i.e., strided
    // (1, x, ..., x, 2, x, ..., x, 3, x, ..., x, ...)
    // At this stage, layout inference would annotate this with #strided attr
    // CHECK: [[to_convert:%.+]] = linalg.reduce
    // CHECK-SAME: {tensor_ext.layout = [#tensor_ext.layout<map = (d0) -> (d0 * 32)>]
    %2 = linalg.reduce { arith.addi } ins(%pt_arg1:!tensor) outs(%out_2:!tensor2) dimensions = [1]

    // CHECK: [[converted:%.+]] = tensor_ext.convert_layout [[to_convert]]
    // CHECK-SAME: from_layout = #tensor_ext.layout<map = (d0) -> (d0 * 32)>
    // CHECK-SAME: tensor_ext.layout = [
    // CHECK-SAME: #tensor_ext.layout<map = (d0) -> (d0)>
    // CHECK-SAME: to_layout = #tensor_ext.layout<map = (d0) -> (d0)>
    // CHECK: arith.addi [[unconverted]], [[converted]]
    %3 = arith.addi %1, %2 : !tensor2
    secret.yield %3 : !tensor2
  } -> !stensor2
  return %0 : !stensor2
}
