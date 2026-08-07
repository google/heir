// RUN: heir-opt --populate-scale-ckks --validate-scale %s | FileCheck %s

#layout = #tensor_ext.layout<"{ [i0, i1] -> [ct, slot] : i0 = 0 and ct = 0 and (-i1 + slot) mod 16 = 0 and 0 <= i1 <= 12 and 0 <= slot <= 4095 }">
#original_type = #tensor_ext.original_type<originalType = tensor<1x13xf32>, layout = #layout>
module attributes {backend.lattigo, ckks.schemeParam = #ckks.scheme_param<logN = 16, Q = [36028797014376449, 1093533697], P = [1152921504614055937, 1152921504615628801], logDefaultScale = 30, encryptionTechnique = extended>, scheme.actual_slot_count = 32768 : i64, scheme.ckks, scheme.requested_slot_count = 8192 : i64} {
  // CHECK: func.func @test_minimal_loop2
  func.func @test_minimal_loop2(%arg0: !secret.secret<tensor<1x4096xf32>> {mgmt.mgmt = #mgmt.mgmt<level = 0>}) -> (!secret.secret<tensor<1x4096xf32>> {mgmt.mgmt = #mgmt.mgmt<level = 0>}) {
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %cst = arith.constant dense<1.0> : tensor<1x4096xf32>

    %0 = secret.generic(%arg0: !secret.secret<tensor<1x4096xf32>>) {
    ^body(%input0: tensor<1x4096xf32>):
      // A loop at level 0
      %loop = scf.for %arg2 = %c1 to %c32 step %c1 iter_args(%arg3 = %input0) -> (tensor<1x4096xf32>) {
        %dummy = arith.addf %arg3, %input0 {mgmt.mgmt = #mgmt.mgmt<level = 0>} : tensor<1x4096xf32>
        scf.yield %dummy : tensor<1x4096xf32>
      } {__argattrs = [{}, {}, {}, {mgmt.mgmt = #mgmt.mgmt<level = 0>}], __resattrs = [{mgmt.mgmt = #mgmt.mgmt<level = 0>}]}

      // Level 0 constant
      %cst_lvl0 = mgmt.init %cst {mgmt.mgmt = #mgmt.mgmt<level = 0>} : tensor<1x4096xf32>

      // Level 0 multiplication
      // CHECK: arith.mulf %{{.*}}, %{{.*}} {mgmt.mgmt = #mgmt.mgmt<level = 0, scale = 85>
      %mul = arith.mulf %loop, %cst_lvl0 {mgmt.mgmt = #mgmt.mgmt<level = 0>} : tensor<1x4096xf32>

      secret.yield %mul : tensor<1x4096xf32>
    } -> (!secret.secret<tensor<1x4096xf32>> {mgmt.mgmt = #mgmt.mgmt<level = 0>})
    return %0 : !secret.secret<tensor<1x4096xf32>>
  }
}
