// RUN: heir-opt "--secret-insert-mgmt-ckks=after-mul=true before-mul-include-first-mul=false bootstrap-waterline=10 level-budget=2 slot-number=1024" %s | FileCheck %s

// This test was extracted from a larger matvec example. The main issue is
// that, when the first loop iteration is peeled, the initializer is replaced
// by the constant initializer in the else branch of the scf.yield (which just
// naively returns the iter_arg). In some cases, sccp can resolve this issue
// and detect that the if statement will not execute its else statement. But in
// a doubly nested loop (approximated here when the loop condition is computed
// using a func arg), the else cannot be removed. So we require an additional
// mgmt.init to be inserted to ensure that the secretness aligns on both
// branches.

module attributes {backend.lattigo, scheme.ckks} {
  func.func @loop_with_trivial_if_branch(
      %arg0: !secret.secret<tensor<1x1024xf32>>,
      %arg1: tensor<512x1024xf32>,
      %outer_loop_iv: index
  ) -> !secret.secret<tensor<1x1024xf32>> {
    %c23 = arith.constant 23 : index
    // CHECK: [[CST:%[^ ]*]] = arith.constant dense<0.0{{.*}}> : tensor<1x1024xf32>
    // CHECK: else
    // CHECK-NEXT: [[INIT:%[^ ]*]] = mgmt.init [[CST]]
    // CHECK-NEXT: scf.yield [[INIT]]
    %cst = arith.constant dense<0.000000e+00> : tensor<1x1024xf32>
    %c-23 = arith.constant -23 : index
    %c512 = arith.constant 512 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %0 = secret.generic(%arg0: !secret.secret<tensor<1x1024xf32>>) {
    ^body(%input0: tensor<1x1024xf32>):
      %5 = scf.for %arg4 = %c0 to %c23 step %c1 iter_args(%arg5 = %cst) -> (tensor<1x1024xf32>) {
        %9 = arith.muli %outer_loop_iv, %c23 : index
        %10 = arith.addi %arg4, %9 : index
        %11 = arith.cmpi slt, %10, %c512 : index
        %12 = scf.if %11 -> (tensor<1x1024xf32>) {
          %extracted_slice = tensor.extract_slice %arg1[%10, 0] [1, 1024] [1, 1] : tensor<512x1024xf32> to tensor<1x1024xf32>
          %13 = arith.muli %outer_loop_iv, %c-23 : index
          %14 = tensor_ext.rotate %extracted_slice, %13 : tensor<1x1024xf32>, index
          %15 = tensor_ext.rotate %input0, %arg4 : tensor<1x1024xf32>, index
          %16 = arith.mulf %14, %15 : tensor<1x1024xf32>
          %17 = arith.addf %arg5, %16 : tensor<1x1024xf32>
          scf.yield %17 : tensor<1x1024xf32>
        } else {
          scf.yield %arg5 : tensor<1x1024xf32>
        }
        scf.yield %12 : tensor<1x1024xf32>
      }
      %6 = arith.muli %outer_loop_iv, %c23 : index
      %7 = tensor_ext.rotate %5, %6 : tensor<1x1024xf32>, index
      secret.yield %7 : tensor<1x1024xf32>
    } -> !secret.secret<tensor<1x1024xf32>>
    return %0 : !secret.secret<tensor<1x1024xf32>>
  }
}
