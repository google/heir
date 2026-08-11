// RUN: heir-opt --secret-insert-mgmt-ckks="after-mul=true before-mul-include-first-mul=false bootstrap-waterline=11 level-budget=11 min-slot-count=4096" %s | FileCheck %s

module attributes {backend.lattigo, scheme.ckks} {
  // CHECK: @test_slice_hoist
  func.func @test_slice_hoist(
    %init_val: !secret.secret<tensor<1x4096xf32>>,
    %invariant_tensor: !secret.secret<tensor<23x4096xf32>>
  ) -> !secret.secret<tensor<1x4096xf32>> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c23 = arith.constant 23 : index

    // CHECK: secret.generic
    // CHECK-SAME: {
    // CHECK:      ^body(%[[INIT_TENSOR:.*]]: tensor<1x4096xf32>, %[[INV_TENSOR:.*]]: tensor<23x4096xf32>):
    // CHECK:        %[[BOOTSTRAPPED_INV:.*]] = mgmt.bootstrap %[[INV_TENSOR]]
    // CHECK:        scf.for %[[IDX:.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ITER_ARG:.*]] = %{{.*}}) -> (tensor<1x4096xf32>) {
    // CHECK:          %[[BOOTSTRAPPED_ITER:.*]] = mgmt.bootstrap %[[ITER_ARG]]
    // CHECK:          %[[INIT_IDX:.*]] = mgmt.init %[[IDX]]
    // CHECK:          %[[SLICE:.*]] = tensor.extract_slice %[[BOOTSTRAPPED_INV]][%[[INIT_IDX]], 0] [1, 4096] [1, 1] {{.*}} : tensor<23x4096xf32> to tensor<1x4096xf32>
    // CHECK:          arith.mulf %[[BOOTSTRAPPED_ITER]], %[[SLICE]]
    %result = secret.generic
       (%init_val: !secret.secret<tensor<1x4096xf32>>,
        %invariant_tensor: !secret.secret<tensor<23x4096xf32>>) {
      ^body(%init_tensor: tensor<1x4096xf32>, %inv_tensor: tensor<23x4096xf32>):

        %out_loop = scf.for %arg1 = %c0 to %c23 step %c1 iter_args(%arg2_iter = %init_tensor) -> (tensor<1x4096xf32>) {
          %slice = tensor.extract_slice %inv_tensor[%arg1, 0] [1, 4096] [1, 1] : tensor<23x4096xf32> to tensor<1x4096xf32>
          %mul = arith.mulf %arg2_iter, %slice : tensor<1x4096xf32>
          scf.yield %mul : tensor<1x4096xf32>
        }
        secret.yield %out_loop : tensor<1x4096xf32>
      } -> (!secret.secret<tensor<1x4096xf32>>)
    return %result : !secret.secret<tensor<1x4096xf32>>
  }

  // CHECK: @test_extract_hoist
  func.func @test_extract_hoist(
    %init_val: !secret.secret<f32>,
    %invariant_tensor: !secret.secret<tensor<23xf32>>
  ) -> !secret.secret<f32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c23 = arith.constant 23 : index

    // CHECK: secret.generic
    // CHECK-SAME: {
    // CHECK:      ^body(%[[INIT_VAL:.*]]: f32, %[[INV_TENSOR:.*]]: tensor<23xf32>):
    // CHECK:        %[[BOOTSTRAPPED_INV:.*]] = mgmt.bootstrap %[[INV_TENSOR]]
    // CHECK:        scf.for %[[IDX:.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ITER_ARG:.*]] = %{{.*}}) -> (f32) {
    // CHECK:          %[[BOOTSTRAPPED_ITER:.*]] = mgmt.bootstrap %[[ITER_ARG]]
    // CHECK:          %[[VAL:.*]] = tensor.extract %[[BOOTSTRAPPED_INV]][%[[IDX]]] {{.*}} : tensor<23xf32>
    // CHECK:          arith.mulf %[[BOOTSTRAPPED_ITER]], %[[VAL]]
    %result = secret.generic
       (%init_val: !secret.secret<f32>,
        %invariant_tensor: !secret.secret<tensor<23xf32>>) {
      ^body(%init_tensor: f32, %inv_tensor: tensor<23xf32>):

        %out_loop = scf.for %arg1 = %c0 to %c23 step %c1 iter_args(%arg2_iter = %init_tensor) -> (f32) {
          %val = tensor.extract %inv_tensor[%arg1] : tensor<23xf32>
          %mul = arith.mulf %arg2_iter, %val : f32
          scf.yield %mul : f32
        }
        secret.yield %out_loop : f32
      } -> (!secret.secret<f32>)
    return %result : !secret.secret<f32>
  }
}
