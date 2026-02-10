// RUN: heir-opt --secret-insert-mgmt-ckks="slot-number=8 level-budget=4 after-mul=false before-mul-include-first-mul=true" %s | FileCheck %s

module attributes {backend.lattigo, scheme.ckks} {
  // CHECK: func.func @loop_mul
  // CHECK-SAME: #mgmt.mgmt<level = 4>
  // CHECK: affine.for {{[^ ]*}} = 1 to 9 step 4
  func.func @loop_mul(%arg0: !secret.secret<tensor<8xf32>>) -> !secret.secret<tensor<8xf32>> {
  %c2 = arith.constant dense<2.0> : tensor<8xf32>
  %0 = secret.generic(%arg0: !secret.secret<tensor<8xf32>>) {
  ^body(%arg0_val: tensor<8xf32>):
    %res = affine.for %i = 0 to 10 iter_args(%sum_iter = %c2) -> tensor<8xf32> {
      %sum = arith.mulf %sum_iter, %arg0_val : tensor<8xf32>
      affine.yield %sum : tensor<8xf32>
    }
    secret.yield %res : tensor<8xf32>
  } -> !secret.secret<tensor<8xf32>>
  return %0 : !secret.secret<tensor<8xf32>>
  }
}
