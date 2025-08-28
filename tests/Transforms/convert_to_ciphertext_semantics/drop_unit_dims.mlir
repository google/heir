// RUN: heir-opt %s --convert-to-ciphertext-semantics --split-input-file | FileCheck %s

#kernel = #secret.kernel<name = "MatvecDiagonal", force = false>
#new_layout1 = #tensor_ext.new_layout<"{ [i0, i1] -> [ct, slot] : (i0 - i1 + ct) mod 512 = 0 and (-i1 + ct + slot) mod 1024 = 0 and 0 <= i0 <= 511 and 0 <= i1 <= 783 and 0 <= ct <= 511 and 0 <= slot <= 1023 }">
#new_layout2 = #tensor_ext.new_layout<"{ [i0] -> [ct, slot] : ct = 0 and (-i0 + slot) mod 512 = 0 and 0 <= i0 <= 511 and 0 <= slot <= 1023 }">
#new_layout5 = #tensor_ext.new_layout<"{ [i0, i1] -> [ct, slot] : i0 = 0 and ct = 0 and (-i1 + slot) mod 1024 = 0 and 0 <= i1 <= 783 and 0 <= slot <= 1023 }">
#new_layout6 = #tensor_ext.new_layout<"{ [i0] -> [ct, slot] : ct = 0 and (-i0 + slot) mod 1024 = 0 and 0 <= i0 <= 783 and 0 <= slot <= 1023 }">
module{
  // CHECK: func.func @main
  func.func @main(%arg0: tensor<512x784xf32>, %arg1: tensor<512xf32>, %arg4: !secret.secret<tensor<1x784xf32>> {tensor_ext.layout = #new_layout5}) -> (!secret.secret<tensor<512xf32>> {jax.result_info = "result[0]", tensor_ext.layout = #new_layout2}) {
    %cst = arith.constant dense<0.000000e+00> : tensor<512xf32>
    %0 = tensor_ext.assign_layout %arg0 {layout = #new_layout1, tensor_ext.layout = #new_layout1} : tensor<512x784xf32>
    %1 = tensor_ext.assign_layout %cst {layout = #new_layout2, tensor_ext.layout = #new_layout2} : tensor<512xf32>
    %2 = tensor_ext.assign_layout %arg1 {layout = #new_layout2, tensor_ext.layout = #new_layout2} : tensor<512xf32>
    // CHECK: secret.generic(%[[arg2:.*]]: !secret.secret<tensor<1x1024xf32>>)
    // CHECK: ^body(%[[input0:.*]]: tensor<1x1024xf32>)
    // CHECK: %[[collapsed:.*]] = tensor.collapse_shape %[[input0]]
    // CHECK-SAME: tensor<1x1024xf32> into tensor<1024xf32>
    // CHECK: %[[v4:.*]] = tensor_ext.rotate_and_reduce %[[collapsed]]
    // CHECK: %[[collapsed_2:.*]] = tensor.collapse_shape
    // CHECK: %[[v5:.*]] = arith.addf %[[v4]], %[[collapsed_2]] : tensor<1024xf32>
    // CHECK: %[[v6:.*]] = tensor_ext.rotate %[[v5]], %[[c512:.*]] : tensor<1024xf32>
    // CHECK: %[[expanded:.*]] = tensor.expand_shape
    // CHECK-SAME: tensor<1024xf32> into tensor<1x1024xf32>
    // CHECK: secret.yield %[[expanded]]
    %7 = secret.generic(%arg4: !secret.secret<tensor<1x784xf32>> {tensor_ext.layout = #new_layout5}) {
    ^body(%input0: tensor<1x784xf32>):
      %collapsed = tensor.collapse_shape %input0 [[0, 1]] {tensor_ext.layout = #new_layout6} : tensor<1x784xf32> into tensor<784xf32>
      %8 = linalg.matvec {secret.kernel = #kernel, tensor_ext.layout = #new_layout2} ins(%0, %collapsed : tensor<512x784xf32>, tensor<784xf32>) outs(%1 : tensor<512xf32>) -> tensor<512xf32>
      %9 = arith.addf %2, %8 {tensor_ext.layout = #new_layout2} : tensor<512xf32>
      secret.yield %9 : tensor<512xf32>
    } -> (!secret.secret<tensor<512xf32>> {tensor_ext.layout = #new_layout2})
    return %7 : !secret.secret<tensor<512xf32>>
  }
}
