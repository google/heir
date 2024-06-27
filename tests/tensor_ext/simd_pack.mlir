// RUN: heir-opt --secretize=entry-function=add --wrap-generic --canonicalize --cse \
// RUN:   --align-tensor-sizes %s | FileCheck %s

module  {
  // CHECK-LABEL: @add
  // CHECK-SAME{LITERAL}: !secret.secret<tensor<1024xi16, #tensor_ext.simd_packing<in = [16], padding = [0], out = [1024]>>>
  func.func @add(%arg0: tensor<16xi16>, %arg1: tensor<16xi16>) -> tensor<16xi16> {
    // CHECK-NEXT: secret.generic ins(%[[arg0:.*]], %[[arg1:.*]] : !secret.secret<[[ty:.*]]>, !secret.secret<[[ty]]>)
    // CHECK-NEXT:  ^bb0(%[[arg0_0:.*]]: [[ty]], %[[arg1_0:.*]]: [[ty]]):
    // CHECK-NEXT:    %[[add:.*]] = arith.addi %[[arg0_0]], %[[arg1_0]] : [[ty]]
    // CHECK-NEXT:    secret.yield %[[add]] : [[ty]]
    %0 = arith.addi %arg0, %arg1 : tensor<16xi16>
    return %0 : tensor<16xi16>
  }
}
