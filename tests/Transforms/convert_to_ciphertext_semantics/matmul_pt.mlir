// RUN: heir-opt --convert-to-ciphertext-semantics %s | FileCheck %s

#kernel = #secret.kernel<name = "MatmulBicyclicDiagonal", force = false>

#layout_ct = #tensor_ext.layout<"{ [i0, i1] -> [ct, slot] : ct = 0 and (5i0 - 6i1 + slot) mod 15 = 0 and 0 <= i0 <= 2 and 0 <= i1 <= 4 and 0 <= slot <= 63 }">
#layout_pt = #tensor_ext.layout<"{ [i0, i1] -> [ct, slot] : (14i0 - 15i1 - 7ct + slot) mod 35 = 0 and 0 <= i0 <= 4 and 0 <= i1 <= 6 and 0 <= ct <= 4 and 0 <= slot <= 63 }">
#layout_out = #tensor_ext.layout<"{ [i0, i1] -> [ct, slot] : ct = 0 and (-7i0 + 6i1 + slot) mod 21 = 0 and 0 <= i0 <= 2 and 0 <= i1 <= 6 and 0 <= slot <= 63 }">

#layout_pt2 = #tensor_ext.layout<"{ [i0, i1] -> [ct, slot] : (5i0 - 6i1 - 3ct + slot) mod 15 = 0 and 0 <= i0 <= 2 and 0 <= i1 <= 4 and 0 <= ct <= 4 and 0 <= slot <= 63 }">
#layout_ct2 = #tensor_ext.layout<"{ [i0, i1] -> [ct, slot] : ct = 0 and (14i0 - 15i1 + slot) mod 35 = 0 and 0 <= i0 <= 4 and 0 <= i1 <= 6 and 0 <= slot <= 63 }">

module {
  // CHECK: @matmul_ctpt
  // CHECK-NOT: linalg.matmul
  // CHECK: tensor_ext.rotate
  // CHECK: arith.mulf
  func.func @matmul_ctpt(%arg0: !secret.secret<tensor<3x5xf32>> {tensor_ext.layout = #layout_ct}, %arg1: tensor<5x7xf32>) -> (!secret.secret<tensor<3x7xf32>> {tensor_ext.layout = #layout_out}) {
    %cst = arith.constant dense<0.000000e+00> : tensor<3x7xf32>
    %0 = secret.generic(%arg0: !secret.secret<tensor<3x5xf32>> {tensor_ext.layout = #layout_ct}) {
    ^body(%input0: tensor<3x5xf32>):
      %1 = tensor_ext.assign_layout %arg1 {layout = #layout_pt, tensor_ext.layout = #layout_pt} : tensor<5x7xf32>
      %2 = tensor_ext.assign_layout %cst {layout = #layout_out, tensor_ext.layout = #layout_out} : tensor<3x7xf32>
      %3 = linalg.matmul {secret.kernel = #kernel, tensor_ext.layout = #layout_out} ins(%input0, %1 : tensor<3x5xf32>, tensor<5x7xf32>) outs(%2 : tensor<3x7xf32>) -> tensor<3x7xf32>
      secret.yield %3 : tensor<3x7xf32>
    } -> (!secret.secret<tensor<3x7xf32>> {tensor_ext.layout = #layout_out})
    return %0 : !secret.secret<tensor<3x7xf32>>
  }

  // CHECK: @matmul_ptct
  // CHECK-NOT: linalg.matmul
  // CHECK: tensor_ext.rotate
  // CHECK: arith.mulf
  func.func @matmul_ptct(%arg0: tensor<3x5xf32>, %arg1: !secret.secret<tensor<5x7xf32>> {tensor_ext.layout = #layout_ct2}) -> (!secret.secret<tensor<3x7xf32>> {tensor_ext.layout = #layout_out}) {
    %cst = arith.constant dense<0.000000e+00> : tensor<3x7xf32>
    %0 = secret.generic(%arg1: !secret.secret<tensor<5x7xf32>> {tensor_ext.layout = #layout_ct2}) {
    ^body(%input0: tensor<5x7xf32>):
      %1 = tensor_ext.assign_layout %arg0 {layout = #layout_pt2, tensor_ext.layout = #layout_pt2} : tensor<3x5xf32>
      %2 = tensor_ext.assign_layout %cst {layout = #layout_out, tensor_ext.layout = #layout_out} : tensor<3x7xf32>
      %3 = linalg.matmul {secret.kernel = #kernel, tensor_ext.layout = #layout_out} ins(%1, %input0 : tensor<3x5xf32>, tensor<5x7xf32>) outs(%2 : tensor<3x7xf32>) -> tensor<3x7xf32>
      secret.yield %3 : tensor<3x7xf32>
    } -> (!secret.secret<tensor<3x7xf32>> {tensor_ext.layout = #layout_out})
    return %0 : !secret.secret<tensor<3x7xf32>>
  }
}
