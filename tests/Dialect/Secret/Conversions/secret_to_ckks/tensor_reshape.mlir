// RUN: heir-opt %s --secret-to-ckks | FileCheck %s

// CHECK: ![[ct:.*]] = !lwe.lwe_ciphertext
#new_layout = #tensor_ext.new_layout<"{ [i0, i1] -> [ct, slot] : i0 = 0 and ct = 0 and (-i1 + slot) mod 16 = 0 and 0 <= i1 <= 9 and 0 <= slot <= 1023 }">
#original_type = #tensor_ext.original_type<originalType = tensor<1x10xf32>, layout = #new_layout>
module attributes {ckks.schemeParam = #ckks.scheme_param<logN = 13, Q = [36028797019389953], P = [36028797019488257], logDefaultScale = 45>, jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32, scheme.ckks} {
  // CHECK: func.func @main
  // CHECK-SAME: %[[arg0:.*]]: tensor<1x![[ct]]>
  // CHECK: %[[c0:.*]] = arith.constant 0 : index
  func.func @main(%arg0: !secret.secret<tensor<1x1024xf32>> {mgmt.mgmt = #mgmt.mgmt<level = 0, scale = 45>, tensor_ext.original_type = #tensor_ext.original_type<originalType = tensor<1x784xf32>, layout = #tensor_ext.new_layout<"{ [i0, i1] -> [ct, slot] : i0 = 0 and ct = 0 and (-i1 + slot) mod 1024 = 0 and 0 <= i1 <= 783 and 0 <= slot <= 1023 }">>}) -> (!secret.secret<tensor<1x1024xf32>> {jax.result_info = "result[0]", mgmt.mgmt = #mgmt.mgmt<level = 0, scale = 45>, tensor_ext.original_type = #original_type}) {
    // CHECK: %[[extracted:.*]] = tensor.extract %[[arg0]][%[[c0]]] : tensor<1x![[ct]]>
    // CHECK: %[[v0:.*]] = ckks.add %[[extracted]], %[[extracted]] : (![[ct]], ![[ct]]) -> ![[ct]]
    // CHECK: %[[from_elements:.*]] = tensor.from_elements %[[v0]] : tensor<1x![[ct]]>
    // CHECK: return %[[from_elements]] : tensor<1x![[ct]]>
    %0 = secret.generic(%arg0: !secret.secret<tensor<1x1024xf32>>) {
    ^body(%input0: tensor<1x1024xf32>):
      %collapsed = tensor.collapse_shape %input0 [[0, 1]] : tensor<1x1024xf32> into tensor<1024xf32>
      secret.yield %collapsed : tensor<1024xf32>
    } -> (!secret.secret<tensor<1024xf32>> {mgmt.mgmt = #mgmt.mgmt<level = 0, scale = 45>})
    %1 = secret.generic(%0: !secret.secret<tensor<1024xf32>>) {
    ^body(%input0: tensor<1024xf32>):
      %3 = arith.addf %input0, %input0 : tensor<1024xf32>
      secret.yield %3 : tensor<1024xf32>
    } -> (!secret.secret<tensor<1024xf32>> {mgmt.mgmt = #mgmt.mgmt<level = 0, scale = 45>})
    %2 = secret.generic(%1: !secret.secret<tensor<1024xf32>>) {
    ^body(%input0: tensor<1024xf32>):
      %expanded = tensor.expand_shape %input0 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
      secret.yield %expanded : tensor<1x1024xf32>
    } -> (!secret.secret<tensor<1x1024xf32>> {mgmt.mgmt = #mgmt.mgmt<level = 0, scale = 45>})
    return %2 : !secret.secret<tensor<1x1024xf32>>
  }
}
