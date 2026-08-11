// RUN: heir-opt --secret-insert-mgmt-ckks="after-mul=true before-mul-include-first-mul=false bootstrap-waterline=11 level-budget=11 min-slot-count=4096" %s | FileCheck %s

module attributes {backend.lattigo, scheme.ckks} {
  // CHECK: @test_loop_hoist
  func.func @test_loop_hoist(
    %secret_input: !secret.secret<tensor<23x4096xf32>>,
    %init_val: !secret.secret<tensor<1x4096xf32>>,
    %invariant_val: !secret.secret<tensor<1x4096xf32>>
  ) -> !secret.secret<tensor<1x4096xf32>> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c23 = arith.constant 23 : index

    // CHECK: secret.generic
    // CHECK-SAME: {
    // CHECK:      ^body(%[[INIT_TENSOR:.*]]: tensor<1x4096xf32>, %[[INV_TENSOR:.*]]: tensor<1x4096xf32>):
    // CHECK:        %[[BOOTSTRAPPED_INV:.*]] = mgmt.bootstrap %[[INV_TENSOR]]
    // CHECK:        scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ITER_ARG:.*]] = %{{.*}}) -> (tensor<1x4096xf32>) {
    // CHECK:          %[[BOOTSTRAPPED_ITER:.*]] = mgmt.bootstrap %[[ITER_ARG]]
    // CHECK:          arith.mulf %[[BOOTSTRAPPED_ITER]], %[[BOOTSTRAPPED_INV]]
    %result = secret.generic
      (%secret_input: !secret.secret<tensor<23x4096xf32>>,
       %init_val: !secret.secret<tensor<1x4096xf32>>,
       %invariant_val: !secret.secret<tensor<1x4096xf32>>) {
      ^body(%secret_tensor: tensor<23x4096xf32>, %init_tensor: tensor<1x4096xf32>, %inv_tensor: tensor<1x4096xf32>):

        %out_loop = scf.for %arg1 = %c0 to %c23 step %c1 iter_args(%arg2_iter = %init_tensor) -> (tensor<1x4096xf32>) {
          %mul = arith.mulf %arg2_iter, %inv_tensor : tensor<1x4096xf32>
          scf.yield %mul : tensor<1x4096xf32>
        }
        secret.yield %out_loop : tensor<1x4096xf32>
      } -> (!secret.secret<tensor<1x4096xf32>>)
    return %result : !secret.secret<tensor<1x4096xf32>>
  }
}
