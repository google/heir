// RUN: heir-opt --secret-to-ckks %s | FileCheck %s


// CHECK: ![[ct_L1:.*]] = !lwe.lwe_ciphertext

#layout = #tensor_ext.layout<"{ [i0, i1] -> [ct, slot] : i0 = 0 and ct = 0 and (-i1 + slot) mod 512 = 0 and 0 <= i1 <= 511 and 0 <= slot <= 1023 }">
#layout1 = #tensor_ext.layout<"{ [i0, i1] -> [ct, slot] : (i0 - i1 + ct) mod 512 = 0 and (-i1 + ct + slot) mod 1024 = 0 and 0 <= i0 <= 511 and 0 <= i1 <= 783 and 0 <= ct <= 511 and 0 <= slot <= 1023 }">
#layout2 = #tensor_ext.layout<"{ [i0, i1] -> [ct, slot] : i0 = 0 and ct = 0 and (-i1 + slot) mod 1024 = 0 and 0 <= i1 <= 783 and 0 <= slot <= 1023 }">
#original_type = #tensor_ext.original_type<originalType = tensor<1x512xf32>, layout = #layout>
module attributes {backend.openfhe, ckks.schemeParam = #ckks.scheme_param<logN = 13, Q = [36028797018652673, 35184372121601], P = [1152921504606994433], logDefaultScale = 45>, scheme.ckks}  {
  // CHECK: func.func private @_assign_layout_1368187199173970310(tensor<512x784xf32>) -> tensor<512x1024xf32>
  // CHECK: func.func @mnist__preprocessed
  // CHECK-SAME: %[[ARG0:.*]]: tensor<512x1024xf32> {
  // CHECK-SAME: %[[ARG1:.*]]: tensor<1x![[ct_L1]]>
  func.func private @_assign_layout_1368187199173970310(tensor<512x784xf32>) -> tensor<512x1024xf32> attributes {client.pack_func = {func_name = "mnist"}}
  func.func @mnist__preprocessed(%arg0: tensor<512x1024xf32> {tensor_ext.original_type = #tensor_ext.original_type<originalType = tensor<512x784xf32>, layout = #layout1>}, %arg1: !secret.secret<tensor<1x1024xf32>> {mgmt.mgmt = #mgmt.mgmt<level = 1>, tensor_ext.original_type = #tensor_ext.original_type<originalType = tensor<1x784xf32>, layout = #layout2>}) -> (!secret.secret<tensor<1x1024xf32>> {jax.result_info = "result[0]", mgmt.mgmt = #mgmt.mgmt<level = 1>, tensor_ext.original_type = #original_type}) attributes {client.preprocessed_func = {func_name = "mnist"}} {
    %extracted_slice = tensor.extract_slice %arg0[0, 0] [1, 1024] [1, 1] : tensor<512x1024xf32> to tensor<1x1024xf32>
    %0 = mgmt.init %extracted_slice {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<1x1024xf32>
    %1 = secret.generic(%arg1: !secret.secret<tensor<1x1024xf32>>) {
    ^body(%input0: tensor<1x1024xf32>):
      %2 = arith.mulf %0, %input0 : tensor<1x1024xf32>
      secret.yield %2 : tensor<1x1024xf32>
    } -> (!secret.secret<tensor<1x1024xf32>> {mgmt.mgmt = #mgmt.mgmt<level = 1>})
    return %1 : !secret.secret<tensor<1x1024xf32>>
  }
  // CHECK: func.func public @mnist
  // CHECK-SAME: %[[ARG0:.*]]: tensor<512x784xf32>
  // CHECK-SAME: %[[ARG1:.*]]: tensor<1x![[ct_L1]]>
  // CHECK: %[[v0:.*]] = call @_assign_layout_1368187199173970310(%[[ARG0]])
  // CHECK: %[[v1:.*]] = call @mnist__preprocessed(%[[v0]], %[[ARG1]])
  // CHECK: return %[[v1]] : tensor<1x![[ct_L1]]>
  func.func public @mnist(%arg0: tensor<512x784xf32>, %arg1: !secret.secret<tensor<1x1024xf32>> {mgmt.mgmt = #mgmt.mgmt<level = 1>, tensor_ext.original_type = #tensor_ext.original_type<originalType = tensor<1x784xf32>, layout = #layout2>}) -> (!secret.secret<tensor<1x1024xf32>> {jax.result_info = "result[0]", mgmt.mgmt = #mgmt.mgmt<level = 1>, tensor_ext.original_type = #original_type}) {
    %cst = arith.constant dense<0.000000e+00> : tensor<512xf32>
    %0 = call @_assign_layout_1368187199173970310(%arg0) : (tensor<512x784xf32>) -> tensor<512x1024xf32>
    %1 = call @mnist__preprocessed(%0, %arg1) {__resattrs = [{mgmt.mgmt = #mgmt.mgmt<level = 1>}], arg_attrs = [{tensor_ext.layout = #layout1}, {tensor_ext.layout = #layout2}]} : (tensor<512x1024xf32>, !secret.secret<tensor<1x1024xf32>>) -> !secret.secret<tensor<1x1024xf32>>
    return %1 : !secret.secret<tensor<1x1024xf32>>
  }
}
