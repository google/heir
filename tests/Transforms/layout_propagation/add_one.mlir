// RUN: heir-opt --layout-propagation %s | FileCheck %s

!stensor = !secret.secret<tensor<32xi16>>

// CHECK: elementwise_sum
// CHECK-NOT: convert_layout
func.func @elementwise_sum(%arg0: !stensor, %arg1: !stensor) -> !stensor {
  %0 = secret.generic(%arg0: !stensor, %arg1: !stensor) {
  ^body(%pt_arg0: tensor<32xi16>, %pt_arg1: tensor<32xi16>):
    %3 = arith.addi %pt_arg0, %pt_arg1: tensor<32xi16>
    secret.yield %3 : tensor<32xi16>
  } -> !stensor
  return %0 : !stensor
}
