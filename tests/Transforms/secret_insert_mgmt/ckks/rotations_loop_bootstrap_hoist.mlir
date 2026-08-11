// RUN: heir-opt --secret-insert-mgmt-ckks="after-mul=true before-mul-include-first-mul=false bootstrap-waterline=11 level-budget=11 min-slot-count=4096" %s | FileCheck %s

module attributes {backend.lattigo, scheme.ckks} {
  // CHECK: @test_rotations_loop_hoist
  func.func @test_rotations_loop_hoist(
    %init_val: !secret.secret<tensor<1x4096xf32>>,
    %input_val: !secret.secret<tensor<1x4096xf32>>
  ) -> !secret.secret<tensor<1x4096xf32>> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c23 = arith.constant 23 : index

    // CHECK: secret.generic
    // CHECK-SAME: {
    // CHECK:      ^body(%[[INIT_TENSOR:.*]]: tensor<1x4096xf32>, %[[INPUT_TENSOR:.*]]: tensor<1x4096xf32>):
    // CHECK:        %[[BOOT_INPUT:.*]] = mgmt.bootstrap %[[INPUT_TENSOR]]
    // CHECK:        %[[INIT_SLICE:.*]] = tensor.insert_slice %[[BOOT_INPUT]] into %{{.*}}
    // CHECK:        %[[ROTATIONS:.*]] = scf.for %[[IDX:.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ARG:.*]] = %[[INIT_SLICE]])
    // CHECK:          %[[ROT:.*]] = tensor_ext.rotate %[[BOOT_INPUT]], %[[IDX]]
    // CHECK:          %[[INS:.*]] = tensor.insert_slice %[[ROT]] into %[[ARG]]
    // CHECK:          scf.yield %[[INS]]
    // CHECK:        }
    // CHECK:        scf.for %[[IDX2:.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ITER_ARG:.*]] = %{{.*}})
    // CHECK:          %[[BOOT_ITER:.*]] = mgmt.bootstrap %[[ITER_ARG]]
    // CHECK:          %[[INIT_IDX:.*]] = mgmt.init %[[IDX2]]
    // CHECK:          %[[SLICE:.*]] = tensor.extract_slice %[[ROTATIONS]][%[[INIT_IDX]], 0]
    // CHECK:          arith.mulf %[[BOOT_ITER]], %[[SLICE]]
    %result = secret.generic
       (%init_val: !secret.secret<tensor<1x4096xf32>>,
        %input_val: !secret.secret<tensor<1x4096xf32>>) {
      ^body(%init_tensor: tensor<1x4096xf32>, %input_tensor: tensor<1x4096xf32>):
        %empty = tensor.empty() : tensor<23x4096xf32>
        %init_slice = tensor.insert_slice %input_tensor into %empty[0, 0] [1, 4096] [1, 1] : tensor<1x4096xf32> into tensor<23x4096xf32>
        %rotations = scf.for %arg1 = %c1 to %c23 step %c1 iter_args(%arg2 = %init_slice) -> (tensor<23x4096xf32>) {
          %rot = tensor_ext.rotate %input_tensor, %arg1 : tensor<1x4096xf32>, index
          %inserted = tensor.insert_slice %rot into %arg2[%arg1, 0] [1, 4096] [1, 1] : tensor<1x4096xf32> into tensor<23x4096xf32>
          scf.yield %inserted : tensor<23x4096xf32>
        }

        %out_loop = scf.for %arg1 = %c0 to %c23 step %c1 iter_args(%arg2_iter = %init_tensor) -> (tensor<1x4096xf32>) {
          %slice = tensor.extract_slice %rotations[%arg1, 0] [1, 4096] [1, 1] : tensor<23x4096xf32> to tensor<1x4096xf32>
          %mul = arith.mulf %arg2_iter, %slice : tensor<1x4096xf32>
          scf.yield %mul : tensor<1x4096xf32>
        }
        secret.yield %out_loop : tensor<1x4096xf32>
      } -> (!secret.secret<tensor<1x4096xf32>>)
    return %result : !secret.secret<tensor<1x4096xf32>>
  }
}
