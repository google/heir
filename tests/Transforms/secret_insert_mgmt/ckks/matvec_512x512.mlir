// RUN: heir-opt --secret-insert-mgmt-ckks="level-budget=40 after-mul=true" %s | FileCheck %s

// CHECK: @matvec
// CHECK-NOT: mgmt.bootstrap
// CHECK: scf.for
// CHECK: scf.for
// CHECK: return

#layout = #tensor_ext.layout<"{ [i0] -> [ct, slot] : ct = 0 and (-i0 + slot) mod 512 = 0 and 0 <= i0 <= 511 and 0 <= slot <= 1023 }">
#original_type = #tensor_ext.original_type<originalType = tensor<512xf32>, layout = #layout>
module {
  func.func @matvec(%arg0: !secret.secret<tensor<1x1024xf32>> {tensor_ext.original_type = #original_type}) -> (!secret.secret<tensor<1x1024xf32>> {tensor_ext.original_type = #original_type}) {
    %cst = arith.constant dense<0.000000e+00> : tensor<1x1024xf32>
    %c0 = arith.constant 0 : index
    %c23 = arith.constant 23 : index
    %c1 = arith.constant 1 : index
    %c512 = arith.constant 512 : index
    %0 = secret.generic(%arg0: !secret.secret<tensor<1x1024xf32>>) {
    ^body(%input0: tensor<1x1024xf32>):
      %1 = scf.for %arg1 = %c0 to %c23 step %c1 iter_args(%arg2 = %cst) -> (tensor<1x1024xf32>) {
        %2 = scf.for %arg3 = %c0 to %c23 step %c1 iter_args(%arg4 = %cst) -> (tensor<1x1024xf32>) {
          %6 = arith.muli %arg1, %c23 : index
          %7 = arith.addi %arg3, %6 : index
          %8 = arith.cmpi slt, %7, %c512 : index
          %9 = scf.if %8 -> (tensor<1x1024xf32>) {
            %10 = tensor_ext.rotate %input0, %arg3 : tensor<1x1024xf32>, index
            %11 = arith.addf %arg4, %10 : tensor<1x1024xf32>
            scf.yield %11 : tensor<1x1024xf32>
          } else {
            scf.yield %arg4 : tensor<1x1024xf32>
          }
          scf.yield %9 : tensor<1x1024xf32>
        }
        %3 = arith.muli %arg1, %c23 : index
        %4 = tensor_ext.rotate %2, %3 : tensor<1x1024xf32>, index
        %5 = arith.addf %arg2, %4 : tensor<1x1024xf32>
        scf.yield %5 : tensor<1x1024xf32>
      }
      secret.yield %1 : tensor<1x1024xf32>
    } -> !secret.secret<tensor<1x1024xf32>>
    return %0 : !secret.secret<tensor<1x1024xf32>>
  }
}
