// RUN: heir-opt --secret-to-ckks=poly-mod-degree=1024 %s | FileCheck %s

#layout = #tensor_ext.layout<"{ [i0] -> [ct, slot] : ct = 0 and (-i0 + slot) mod 512 = 0 and 0 <= i0 <= 511 and 0 <= slot <= 1023 }">
#original_type = #tensor_ext.original_type<originalType = tensor<512xf32>, layout = #layout>
module attributes {backend.lattigo, ckks.schemeParam = #ckks.scheme_param<logN = 13, Q = [36028797018652673, 35184372121601], P = [1152921504606994433], logDefaultScale = 45, encryptionTechnique = extended>, scheme.actual_slot_count = 4096 : i64, scheme.ckks, scheme.requested_slot_count = 1024 : i64} {

  // CHECK: func @lower_conceal
  func.func @lower_conceal() -> (!secret.secret<tensor<1x1024xf32>> {mgmt.mgmt = #mgmt.mgmt<level = 0, scale = 45>, tensor_ext.original_type = #original_type}) {
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<1x1024xf32>
    %0 = mgmt.init %cst_0 {mgmt.mgmt = #mgmt.mgmt<level = 0, scale = 45>} : tensor<1x1024xf32>
    // CHECK: lwe.trivial_encrypt
    %1 = secret.conceal %0 {mgmt.mgmt = #mgmt.mgmt<level = 0, scale = 45>} : tensor<1x1024xf32> -> !secret.secret<tensor<1x1024xf32>>
    // CHECK: return
    return %1 : !secret.secret<tensor<1x1024xf32>>
  }
}
