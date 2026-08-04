// RUN: heir-opt --secret-insert-mgmt-ckks="after-mul=true before-mul-include-first-mul=false bootstrap-waterline=11 level-budget=11 min-slot-count=4096" %s | FileCheck %s

// This test verifies that the secret-insert-mgmt-ckks pass correctly handles
// a Lattigo target with non-zero bootstrap levels (16 levels consumed).
// It verifies that we adjust the budget and waterline correctly, and
// calculate the correct unroll factor (11 in this case, since budget is
// 11 + 16 = 27, level after bootstrap is 11, and levels used in loop is 1).
// The outer loop should be unrolled by 11, resulting in step 11.

module attributes {backend.lattigo, scheme.ckks} {
  // CHECK: @test_loop
  func.func @test_loop(
    %secret_input: !secret.secret<tensor<23x4096xf32>>,
    %init_val: !secret.secret<tensor<1x4096xf32>>
  ) -> !secret.secret<tensor<1x4096xf32>> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c23 = arith.constant 23 : index
    %cst_public = arith.constant dense<1.000000e+00> : tensor<23x4096xf32>
    %cst_zero = arith.constant dense<0.000000e+00> : tensor<1x4096xf32>

    // CHECK: %[[C11:.*]] = arith.constant 11 : index
    // CHECK: secret.generic
    %result = secret.generic
      (%secret_input: !secret.secret<tensor<23x4096xf32>>, %init_val: !secret.secret<tensor<1x4096xf32>>) {
      ^body(%secret_tensor: tensor<23x4096xf32>, %init_tensor: tensor<1x4096xf32>):

        // CHECK: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %[[C11]]
        %out_loop = scf.for %arg1 = %c0 to %c23 step %c1 iter_args(%arg2 = %init_tensor) -> (tensor<1x4096xf32>) {

          %sec_slice_out = tensor.extract_slice %secret_tensor[%arg1, 0] [1, 4096] [1, 1] : tensor<23x4096xf32> to tensor<1x4096xf32>
          %mul_out = arith.mulf %arg2, %sec_slice_out : tensor<1x4096xf32>

          %in_loop = scf.for %arg3 = %c0 to %c23 step %c1 iter_args(%arg4 = %cst_zero) -> (tensor<1x4096xf32>) {
            %pub_slice = tensor.extract_slice %cst_public[%arg3, 0] [1, 4096] [1, 1] : tensor<23x4096xf32> to tensor<1x4096xf32>
            %sec_slice = tensor.extract_slice %secret_tensor[%arg3, 0] [1, 4096] [1, 1] : tensor<23x4096xf32> to tensor<1x4096xf32>
            %mul = arith.mulf %pub_slice, %sec_slice : tensor<1x4096xf32>
            %add = arith.addf %arg4, %mul : tensor<1x4096xf32>
            scf.yield %add : tensor<1x4096xf32>
          }

          %add_out = arith.addf %mul_out, %in_loop : tensor<1x4096xf32>
          scf.yield %add_out : tensor<1x4096xf32>
        }
        secret.yield %out_loop : tensor<1x4096xf32>
      } -> (!secret.secret<tensor<1x4096xf32>>)
    return %result : !secret.secret<tensor<1x4096xf32>>
  }
}
