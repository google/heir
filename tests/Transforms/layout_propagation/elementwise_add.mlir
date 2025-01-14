// RUN: heir-opt --layout-propagation %s | FileCheck %s

!stensor = !secret.secret<tensor<32x32xi16>>
#row_major = affine_map<(i, j) -> (32*i + j)>

// Just test that the layout propagation pass runs, even though no layout
// conversion ops are inserted.
// CHECK-LABEL: elementwise_sum
func.func @elementwise_sum(%arg0: !stensor, %arg1: !stensor) -> !stensor {
  %0 = secret.generic ins(%arg0, %arg1: !stensor, !stensor) {
  ^body(%pt_arg0: tensor<32x32xi16>, %pt_arg1: tensor<32x32xi16>):
    %3 = arith.addi %pt_arg0, %pt_arg1: tensor<32x32xi16>
    secret.yield %3 : tensor<32x32xi16>
  } -> !stensor
  return %0 : !stensor
}
