// RUN: heir-opt --secret-to-ckks --canonicalize --split-input-file %s | FileCheck %s

// CHECK: ![[ct:.*]] = !lwe.lwe_ciphertext
#layout = #tensor_ext.layout<"{ [i0, i1, i2, i3] -> [ct, slot] : i0 = 0 and ct = i1 and (2i2 - i3 + slot) mod 4 = 0 and 0 <= i1 <= 1 and 0 <= i2 <= 1 and 0 <= i3 <= 1 and 0 <= slot <= 1023 }">
#original_type = #tensor_ext.original_type<originalType = tensor<1x2x2x2xf32>, layout = #layout>
module attributes {ckks.schemeParam = #ckks.scheme_param<logN = 13, Q = [36028797019389953, 35184372121601], P = [36028797019488257], logDefaultScale = 45>, scheme.ckks} {
  // CHECK: func.func @main
  // CHECK-SAME: (%[[arg0:.*]]: tensor<2x![[ct]]>
  // CHECK: %[[v0:.*]] = tensor.extract_slice %[[arg0]][0] [1] [1] : tensor<2x![[ct]]> to tensor<1x![[ct]]>
  // CHECK: return %[[v0]] : tensor<1x![[ct]]>
  func.func @main(%arg0: !secret.secret<tensor<2x1024xf32>> {mgmt.mgmt = #mgmt.mgmt<level = 1, scale = 90>, tensor_ext.original_type = #tensor_ext.original_type<originalType = tensor<1x1x4x4xf32>, layout = #tensor_ext.layout<"{ [i0, i1, i2, i3] -> [ct, slot] : i0 = 0 and i1 = 0 and ct = 0 and (-4i2 - i3 + slot) mod 16 = 0 and 0 <= i2 <= 3 and 0 <= i3 <= 3 and 0 <= slot <= 1023 }">>}, %arg1: tensor<2x1x3x3xf32>) -> (!secret.secret<tensor<1x1024xf32>> {mgmt.mgmt = #mgmt.mgmt<level = 1, scale = 90>, tensor_ext.original_type = #original_type}) {
    %43 = secret.generic(%arg0: !secret.secret<tensor<2x1024xf32>>) {
    ^body(%input0: tensor<2x1024xf32>):
      %extracted_slice_7 = tensor.extract_slice %input0[0, 0] [1, 1024] [1, 1] : tensor<2x1024xf32> to tensor<1x1024xf32>
      secret.yield %extracted_slice_7 : tensor<1x1024xf32>
    } -> (!secret.secret<tensor<1x1024xf32>> {mgmt.mgmt = #mgmt.mgmt<level = 1, scale = 90>})
    return %43 : !secret.secret<tensor<1x1024xf32>>
  }
}
