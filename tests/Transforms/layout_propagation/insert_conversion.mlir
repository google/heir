// RUN: heir-opt --layout-propagation %s | FileCheck %s

!tensor = tensor<32x32xi16>
!tensor2 = tensor<32xi16>
!stensor = !secret.secret<!tensor>
!stensor2 = !secret.secret<!tensor2>

// Test that when an operation changes the tensor layour in an incompatible way,
// a layout conversion operation is inserted.
// CHECK-LABEL: insert_conversion
func.func @insert_conversion(%arg0: !stensor, %arg1: !stensor) -> !stensor2 {
  %0 = secret.generic ins(%arg0, %arg1: !stensor, !stensor) {
  ^body(%pt_arg0: !tensor, %pt_arg1: !tensor):
    // result of sum has row-major layout, i.e., with don't cares at the end
    // (1, 2, ..., 32, x, x, x, x, ..., x)
    %1 = tensor_ext.sum %pt_arg0, 0 : !tensor -> !tensor2

    // result of sum has column-major layout, i.e., strided
    // (1, x, ..., x, 2, x, ..., x, 3, x, ..., x, ...)
    // At this stage, layout inference would annotate this with #strided attr
    %2 = tensor_ext.sum %pt_arg1, 1 : !tensor -> !tensor2

    %3 = arith.addi %1, %2 : !tensor2
    secret.yield %3 : !tensor2
  } -> !stensor2
  return %0 : !stensor2
}
