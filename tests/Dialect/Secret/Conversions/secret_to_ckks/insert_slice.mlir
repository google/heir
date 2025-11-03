// RUN: heir-opt --secret-to-ckks --canonicalize --split-input-file %s | FileCheck %s

// CHECK: ![[ct:.*]] = !lwe.lwe_ciphertext

#layout = #tensor_ext.layout<"{ [i0, i1, i2, i3] -> [ct, slot] : i0 = 0 and ct = i1 and (2i2 - i3 + slot) mod 4 = 0 and 0 <= i1 <= 1 and 0 <= i2 <= 1 and 0 <= i3 <= 1 and 0 <= slot <= 1023 }">
#original_type = #tensor_ext.original_type<originalType = tensor<1x2x2x2xf32>, layout = #layout>
module attributes {ckks.schemeParam = #ckks.scheme_param<logN = 13, Q = [36028797019389953, 35184372121601], P = [36028797019488257], logDefaultScale = 45>, scheme.ckks} {
  func.func @scalar(%arg0: !secret.secret<tensor<1024xf32>> {mgmt.mgmt = #mgmt.mgmt<level = 1, scale = 45>, tensor_ext.original_type = #tensor_ext.original_type<originalType = tensor<1x1x4x4xf32>, layout = #tensor_ext.layout<"{ [i0, i1, i2, i3] -> [ct, slot] : i0 = 0 and i1 = 0 and ct = 0 and (-4i2 - i3 + slot) mod 16 = 0 and 0 <= i2 <= 3 and 0 <= i3 <= 3 and 0 <= slot <= 1023 }">>}, %arg1: tensor<2x1x3x3xf32>) -> (!secret.secret<tensor<1x1024xf32>> {mgmt.mgmt = #mgmt.mgmt<level = 1, scale = 45>, tensor_ext.original_type = #original_type}) {
    // CHECK: func.func @scalar
    // CHECK-SAME: (%[[arg0:.*]]: ![[ct]]
    // CHECK: %[[v0:.*]] = tensor.empty() : tensor<1x![[ct]]>
    // CHECK: %[[v1:.*]] = tensor.insert %[[arg0]] into %[[v0]]
    // CHECK: return %[[v1]] : tensor<1x![[ct]]>
    %3 = tensor.empty() : tensor<1x1024xf32>
    %5 = mgmt.init %3 {mgmt.mgmt = #mgmt.mgmt<level = 1, scale = 45>} : tensor<1x1024xf32>
    %11 = secret.generic(%arg0: !secret.secret<tensor<1024xf32>>) {
    ^body(%input0: tensor<1024xf32>):
      %inserted_slice_7 = tensor.insert_slice %input0 into %5[0, 0] [1, 1024] [1, 1] : tensor<1024xf32> into tensor<1x1024xf32>
      secret.yield %inserted_slice_7 : tensor<1x1024xf32>
    } -> (!secret.secret<tensor<1x1024xf32>> {mgmt.mgmt = #mgmt.mgmt<level = 1, scale = 45>})
    return %11 : !secret.secret<tensor<1x1024xf32>>
  }
}

// -----

// CHECK: ![[ct:.*]] = !lwe.lwe_ciphertext

#layout = #tensor_ext.layout<"{ [i0, i1, i2, i3] -> [ct, slot] : i0 = 0 and ct = i1 and (2i2 - i3 + slot) mod 4 = 0 and 0 <= i1 <= 1 and 0 <= i2 <= 1 and 0 <= i3 <= 1 and 0 <= slot <= 1023 }">
#original_type = #tensor_ext.original_type<originalType = tensor<1x2x2x2xf32>, layout = #layout>
module attributes {ckks.schemeParam = #ckks.scheme_param<logN = 13, Q = [36028797019389953, 35184372121601], P = [36028797019488257], logDefaultScale = 45>, scheme.ckks} {
  func.func @tensor(%arg0: !secret.secret<tensor<1x1024xf32>> {mgmt.mgmt = #mgmt.mgmt<level = 1, scale = 90>, tensor_ext.original_type = #tensor_ext.original_type<originalType = tensor<1x1x4x4xf32>, layout = #tensor_ext.layout<"{ [i0, i1, i2, i3] -> [ct, slot] : i0 = 0 and i1 = 0 and ct = 0 and (-4i2 - i3 + slot) mod 16 = 0 and 0 <= i2 <= 3 and 0 <= i3 <= 3 and 0 <= slot <= 1023 }">>}, %arg1: tensor<2x1x3x3xf32>) -> (!secret.secret<tensor<2x1024xf32>> {mgmt.mgmt = #mgmt.mgmt<level = 1, scale = 90>, tensor_ext.original_type = #original_type}) {
    // CHECK: func.func @tensor
    // CHECK-SAME: (%[[arg0:.*]]: tensor<1x![[ct]]>
    // CHECK: %[[v0:.*]] = tensor.empty() : tensor<2x![[ct]]>
    // CHECK: %[[v1:.*]] = tensor.insert_slice %[[arg0]] into %[[v0]][1] [1] [1] : tensor<1x![[ct]]> into tensor<2x![[ct]]>
    // CHECK: return %[[v1]] : tensor<2x![[ct]]>
    %4 = tensor.empty() : tensor<2x1024xf32>
    %9 = mgmt.init %4 {mgmt.mgmt = #mgmt.mgmt<level = 1, scale = 90>} : tensor<2x1024xf32>
    %46 = secret.generic(%arg0: !secret.secret<tensor<1x1024xf32>>) {
    ^body(%input0: tensor<1x1024xf32>):
      %inserted_slice_7 = tensor.insert_slice %input0 into %9[1, 0] [1, 1024] [1, 1] : tensor<1x1024xf32> into tensor<2x1024xf32>
      secret.yield %inserted_slice_7 : tensor<2x1024xf32>
    } -> (!secret.secret<tensor<2x1024xf32>> {mgmt.mgmt = #mgmt.mgmt<level = 1, scale = 90>})
    return %46 : !secret.secret<tensor<2x1024xf32>>
  }
}
