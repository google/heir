// RUN: heir-opt --convert-to-ciphertext-semantics %s | FileCheck %s
#kernel = #secret.kernel<name = "MatmulBicyclic", force = false>
#layout = #tensor_ext.layout<"{ [i0, i1] -> [ct, slot] : ct = 0 and (2i0 + 3i1 + slot) mod 6 = 0 and 0 <= i0 <= 2 and 0 <= i1 <= 1 and 0 <= slot <= 1023 }">
#layout1 = #tensor_ext.layout<"{ [i0, i1] -> [ct, slot] : ct = 0 and (5i0 - 6i1 + slot) mod 15 = 0 and 0 <= i0 <= 2 and 0 <= i1 <= 4 and 0 <= slot <= 1023 }">
#layout2 = #tensor_ext.layout<"{ [i0, i1] -> [ct, slot] : ct = 0 and (4i0 + 5i1 + slot) mod 10 = 0 and 0 <= i0 <= 4 and 0 <= i1 <= 1 and 0 <= slot <= 1023 }">
#layout3 = #tensor_ext.layout<"{ [i0, i1] -> [ct, slot] : ct = 0 and (-5i0 - i1 + slot) mod 16 = 0 and 0 <= i0 <= 2 and 0 <= i1 <= 1023 - 5i0 and i1 <= 4 and 0 <= slot <= 1023 and 1024*floor((-16 + 5i0 + i1)/1024) <= -1024 + 5i0 + i1 }">
module {
  // CHECK: @matmul_secret_secret
  // CHECK-NOT: linalg.matmul
  func.func @matmul_secret_secret(%arg0: !secret.secret<tensor<3x5xf32>> {tensor_ext.layout = #layout1}, %arg1: !secret.secret<tensor<5x2xf32>> {tensor_ext.layout = #layout2}) -> (!secret.secret<tensor<3x2xf32>> {tensor_ext.layout = #layout}) {
    %cst = arith.constant dense<0.000000e+00> : tensor<3x2xf32>
    %0 = secret.generic(%arg0: !secret.secret<tensor<3x5xf32>> {tensor_ext.layout = #layout1}, %arg1: !secret.secret<tensor<5x2xf32>> {tensor_ext.layout = #layout2}) {
    ^body(%input0: tensor<3x5xf32>, %input1: tensor<5x2xf32>):
      %1 = tensor_ext.assign_layout %cst {layout = #layout3, tensor_ext.layout = #layout3} : tensor<3x2xf32>
      %2 = linalg.matmul {secret.kernel = #kernel, tensor_ext.layout = #layout} ins(%input0, %input1 : tensor<3x5xf32>, tensor<5x2xf32>) outs(%1 : tensor<3x2xf32>) -> tensor<3x2xf32>
      secret.yield %2 : tensor<3x2xf32>
    } -> (!secret.secret<tensor<3x2xf32>> {tensor_ext.layout = #layout})
    return %0 : !secret.secret<tensor<3x2xf32>>
  }
}


