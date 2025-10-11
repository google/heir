// RUN: heir-opt %s --convert-to-ciphertext-semantics --split-input-file | FileCheck %s

#layout5 = #tensor_ext.layout<"{ [i0, i1] -> [ct, slot] : i0 = 0 and ct = 0 and (-i1 + slot) mod 1024 = 0 and 0 <= i1 <= 783 and 0 <= slot <= 1023 }">
#layout6 = #tensor_ext.layout<"{ [i0] -> [ct, slot] : ct = 0 and (-i0 + slot) mod 1024 = 0 and 0 <= i0 <= 783 and 0 <= slot <= 1023 }">
module {
  // CHECK: func.func @main
  func.func @main(%arg4: !secret.secret<tensor<1x784xf32>> {tensor_ext.layout = #layout5}) -> (!secret.secret<tensor<784xf32>> {tensor_ext.layout = #layout6}) {
    // CHECK: secret.generic
    // CHECK-NEXT: ^body(%[[input0:.*]]: tensor<1x1024xf32>)
    // CHECK: secret.yield %[[input0]]
    %7 = secret.generic(%arg4: !secret.secret<tensor<1x784xf32>> {tensor_ext.layout = #layout5}) {
    ^body(%input0: tensor<1x784xf32>):
      %collapsed = tensor.collapse_shape %input0 [[0, 1]] {tensor_ext.layout = #layout6} : tensor<1x784xf32> into tensor<784xf32>
      secret.yield %collapsed : tensor<784xf32>
    } -> (!secret.secret<tensor<784xf32>> {tensor_ext.layout = #layout6})
    return %7 : !secret.secret<tensor<784xf32>>
  }
}

// -----

// Check when composed with a matvec

#kernel = #secret.kernel<name = "MatvecDiagonal", force = false>
#layout1 = #tensor_ext.layout<"{ [i0, i1] -> [ct, slot] : (i0 - i1 + ct) mod 512 = 0 and (-i1 + ct + slot) mod 1024 = 0 and 0 <= i0 <= 511 and 0 <= i1 <= 783 and 0 <= ct <= 511 and 0 <= slot <= 1023 }">
#layout2 = #tensor_ext.layout<"{ [i0] -> [ct, slot] : ct = 0 and (-i0 + slot) mod 512 = 0 and 0 <= i0 <= 511 and 0 <= slot <= 1023 }">
#layout5 = #tensor_ext.layout<"{ [i0, i1] -> [ct, slot] : i0 = 0 and ct = 0 and (-i1 + slot) mod 1024 = 0 and 0 <= i1 <= 783 and 0 <= slot <= 1023 }">
#layout6 = #tensor_ext.layout<"{ [i0] -> [ct, slot] : ct = 0 and (-i0 + slot) mod 1024 = 0 and 0 <= i0 <= 783 and 0 <= slot <= 1023 }">
module {
  // CHECK: func.func @main
  func.func @main(%arg0: tensor<512x784xf32>, %arg4: !secret.secret<tensor<1x784xf32>> {tensor_ext.layout = #layout5}) -> (!secret.secret<tensor<512xf32>> {tensor_ext.layout = #layout2}) {
    %cst = arith.constant dense<0.000000e+00> : tensor<512xf32>
    %0 = tensor_ext.assign_layout %arg0 {layout = #layout1, tensor_ext.layout = #layout1} : tensor<512x784xf32>
    %1 = tensor_ext.assign_layout %cst {layout = #layout2, tensor_ext.layout = #layout2} : tensor<512xf32>
    // CHECK: secret.generic
    // CHECK-NEXT: ^body(%[[input0:.*]]: tensor<1x1024xf32>)
    %7 = secret.generic(%arg4: !secret.secret<tensor<1x784xf32>> {tensor_ext.layout = #layout5}) {
    ^body(%input0: tensor<1x784xf32>):
      %collapsed = tensor.collapse_shape %input0 [[0, 1]] {tensor_ext.layout = #layout6} : tensor<1x784xf32> into tensor<784xf32>
      %8 = linalg.matvec {secret.kernel = #kernel, tensor_ext.layout = #layout2} ins(%0, %collapsed : tensor<512x784xf32>, tensor<784xf32>) outs(%1 : tensor<512xf32>) -> tensor<512xf32>
      secret.yield %8 : tensor<512xf32>
    } -> (!secret.secret<tensor<512xf32>> {tensor_ext.layout = #layout2})
    // CHECK: return
    return %7 : !secret.secret<tensor<512xf32>>
  }
}
