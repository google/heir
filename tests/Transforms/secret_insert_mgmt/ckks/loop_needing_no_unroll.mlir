// RUN: heir-opt --secret-insert-mgmt-ckks="level-budget=40 after-mul=true" %s | FileCheck %s

// CHECK: @matvec
// CHECK-DAG: %[[c1:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[c16:.*]] = arith.constant 16 : index
// CHECK-DAG: %[[init0:.*]] = mgmt.init %cst_0 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<1x1024xf32>
// CHECK-DAG: %[[init1:.*]] = mgmt.init %cst {mgmt.mgmt = #mgmt.mgmt<level = 0>} : tensor<1x1024xf32>
// CHECK: %[[generic:.*]] = secret.generic
// CHECK: ^body(%[[input:.*]]: tensor<1x1024xf32>):
// CHECK:   %[[mulf0:.*]] = arith.mulf %[[input]], %[[init0]] {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<1x1024xf32>
// CHECK:   %[[modred0:.*]] = mgmt.modreduce %[[mulf0]] {mgmt.mgmt = #mgmt.mgmt<level = 0>} : tensor<1x1024xf32>
// CHECK:   %[[addf0:.*]] = arith.addf %[[init1]], %[[modred0]] {mgmt.mgmt = #mgmt.mgmt<level = 0>} : tensor<1x1024xf32>
// CHECK:   %[[for:.*]] = scf.for %[[arg1:.*]] = %[[c1]] to %[[c16]] step %[[c1]] iter_args(%[[arg2:.*]] = %[[addf0]]) -> (tensor<1x1024xf32>) {
// CHECK:     %[[rotate:.*]] = tensor_ext.rotate %[[input]], %[[arg1]] {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<1x1024xf32>, index
// CHECK:     %[[init2:.*]] = mgmt.init %cst_0 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<1x1024xf32>
// CHECK:     %[[mulf1:.*]] = arith.mulf %[[rotate]], %[[init2]] {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<1x1024xf32>
// CHECK:     %[[modred1:.*]] = mgmt.modreduce %[[mulf1]] {mgmt.mgmt = #mgmt.mgmt<level = 0>} : tensor<1x1024xf32>
// CHECK:     %[[addf1:.*]] = arith.addf %[[arg2]], %[[modred1]] {mgmt.mgmt = #mgmt.mgmt<level = 0>} : tensor<1x1024xf32>
// CHECK:     scf.yield %[[addf1]] : tensor<1x1024xf32>
// CHECK:   }
// CHECK:   secret.yield %[[for]] : tensor<1x1024xf32>

#layout = #tensor_ext.layout<"{ [i0] -> [ct, slot] : ct = 0 and (-i0 + slot) mod 512 = 0 and 0 <= i0 <= 511 and 0 <= slot <= 1023 }">
#original_type = #tensor_ext.original_type<originalType = tensor<512xf32>, layout = #layout>
module {
  func.func @matvec(%arg0: !secret.secret<tensor<1x1024xf32>> {tensor_ext.original_type = #original_type}) -> (!secret.secret<tensor<1x1024xf32>> {tensor_ext.original_type = #original_type}) {
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %c1 = arith.constant 1 : index
    %cst_zero = arith.constant dense<0.000000e+00> : tensor<1x1024xf32>
    %cst_plain = arith.constant dense<2.000000e+00> : tensor<1x1024xf32>

    %0 = secret.generic(%arg0: !secret.secret<tensor<1x1024xf32>>) {
    ^body(%input0: tensor<1x1024xf32>):
      %1 = scf.for %arg1 = %c0 to %c16 step %c1 iter_args(%arg2 = %cst_zero) -> (tensor<1x1024xf32>) {
        %2 = tensor_ext.rotate %input0, %arg1 : tensor<1x1024xf32>, index
        %3 = arith.mulf %2, %cst_plain : tensor<1x1024xf32>
        %4 = arith.addf %arg2, %3 : tensor<1x1024xf32>
        scf.yield %4 : tensor<1x1024xf32>
      }
      secret.yield %1 : tensor<1x1024xf32>
    } -> !secret.secret<tensor<1x1024xf32>>
    return %0 : !secret.secret<tensor<1x1024xf32>>
  }
}
