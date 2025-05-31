// RUN: heir-opt --layout-propagation --fold-convert-layout-into-assign-layout %s | FileCheck %s

!tensor = tensor<32x32xi16>
!tensor2 = tensor<32xi16>
!stensor = !secret.secret<!tensor>
!stensor2 = !secret.secret<!tensor2>

// Test that when an operation changes the tensor layour in an incompatible way,
// a layout conversion operation is inserted.

// CHECK-DAG: [[alignment1d:[^ ]*]] = #tensor_ext.alignment<in = [32], out = [32]>
// CHECK-DAG: [[alignment2d:[^ ]*]] = #tensor_ext.alignment<in = [32, 32], out = [32, 32]>
// CHECK-DAG: [[rm_layout:[^ ]*]] = #tensor_ext.layout<map = (d0, d1) -> ((d0 * 32 + d1) mod 1024), alignment = [[alignment2d]]>
// CHECK-DAG: [[rm_layout1:[^ ]*]] = #tensor_ext.layout<map = (d0) -> (d0 mod 1024), alignment = [[alignment1d]]>
// CHECK-DAG: [[cm_layout:[^ ]*]] = #tensor_ext.layout<map = (d0) -> ((d0 * 32) mod 1024), alignment = [[alignment1d]]>
// CHECK: insert_conversion
// CHECK-SAME: %[[arg0:[^:]+]]: !secret.secret<tensor<32x32xi16>> {tensor_ext.layout = [[rm_layout]]}
// CHECK-SAME: %[[arg1:[^:]+]]: !secret.secret<tensor<32x32xi16>> {tensor_ext.layout = [[rm_layout]]}
func.func @insert_conversion(%arg0: !stensor, %arg1: !stensor) -> !stensor2 {
  // CHECK: [[cst:%.*]] = arith.constant dense<0>
  %out_1 = arith.constant dense<0> : !tensor2
  %out_2 = arith.constant dense<0> : !tensor2

  // CHECK: secret.generic
  // CHECK-SAME: (%[[arg0]]: !secret.secret<tensor<32x32xi16>> {tensor_ext.layout = [[rm_layout]]},
  // CHECK-SAME: %[[arg1]]: !secret.secret<tensor<32x32xi16>> {tensor_ext.layout = [[rm_layout]]}
  %0 = secret.generic(%arg0: !stensor, %arg1: !stensor) {
  ^body(%pt_arg0: !tensor, %pt_arg1: !tensor):
    // CHECK: tensor_ext.assign_layout [[cst]] {layout = [[rm_layout1]], tensor_ext.layout = [[rm_layout1]]}

    // result of sum has row-major layout
    // (1, 2, ..., 32, ... )
    // CHECK: [[unconverted:[^ ]+]] = linalg.reduce
    // CHECK-SAME: {tensor_ext.layout = [[rm_layout1]]
    %1 = linalg.reduce { arith.addi } ins(%pt_arg0:!tensor) outs(%out_1:!tensor2) dimensions = [0]

    // CHECK: tensor_ext.assign_layout [[cst]]
    // CHECK-SAME: layout = [[cm_layout]]
    // CHECK-SAME: tensor_ext.layout = [[cm_layout]]

    // result of sum has column-major layout, i.e., strided
    // (1, x, ..., x, 2, x, ..., x, 3, x, ..., x, ...)
    // At this stage, layout inference would annotate this with #strided attr
    // CHECK: [[to_convert:%.+]] = linalg.reduce
    // CHECK-SAME: {tensor_ext.layout = [[cm_layout]]
    %2 = linalg.reduce { arith.addi } ins(%pt_arg1:!tensor) outs(%out_2:!tensor2) dimensions = [1]

    // CHECK: [[converted:%.+]] = tensor_ext.convert_layout [[to_convert]]
    // CHECK-SAME: from_layout = [[cm_layout]]
    // CHECK-SAME: tensor_ext.layout = [[rm_layout1]]
    // CHECK-SAME: to_layout = [[rm_layout1]]
    // CHECK: arith.addi [[unconverted]], [[converted]]
    %3 = arith.addi %1, %2 : !tensor2

    // Note this one denotes the layout of the result of the generic op
    // CHECK: secret.yield
    // CHECK-NEXT: -> (!secret.secret<tensor<32xi16>> {tensor_ext.layout = [[rm_layout1]]})
    secret.yield %3 : !tensor2
  } -> !stensor2
  return %0 : !stensor2
}
