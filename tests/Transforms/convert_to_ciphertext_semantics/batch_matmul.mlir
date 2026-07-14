// RUN: heir-opt --convert-to-ciphertext-semantics=ciphertext-size=8192 %s | FileCheck %s
#kernel = #secret.kernel<name = "BatchMatmulTricyclic", force = false>
#layout = #tensor_ext.layout<"{ [i0, i1, i2] -> [ct, slot] : ct = 0 and (357i0 + 84i1 + 272i2 + slot) mod 714 = 0 and 0 <= i0 <= 1 and 0 <= i1 <= 16 and 0 <= i2 <= 20 and 0 <= slot <= 8191 }">
#layout1 = #tensor_ext.layout<"{ [i0, i1, i2] -> [ct, slot] : ct = 0 and (323i0 + 152i1 + 170i2 + slot) mod 646 = 0 and 0 <= i0 <= 1 and 0 <= i1 <= 16 and 0 <= i2 <= 18 and 0 <= slot <= 8191 }">
#layout2 = #tensor_ext.layout<"{ [i0, i1, i2] -> [ct, slot] : ct = 0 and (399i0 - 210i1 - 190i2 + slot) mod 798 = 0 and 0 <= i0 <= 1 and 0 <= i1 <= 18 and 0 <= i2 <= 20 and 0 <= slot <= 8191 }">
#layout3 = #tensor_ext.layout<"{ [i0, i1, i2] -> [ct, slot] : ct = 0 and (-323i0 - 19i1 - i2 + slot) mod 1024 = 0 and 0 <= i0 <= 1 and 0 <= i1 <= 16 and 0 <= i2 <= 8191 - 323i0 - 19i1 and i2 <= 18 and 0 <= slot <= 8191 and 8192*floor((-1024 + 323i0 + 19i1 + i2)/8192) <= -8192 + 323i0 + 19i1 + i2 }">
module {
  // CHECK: @batch_matmul_secret_secret
  // CHECK-NOT: linalg.batch_matmul
  func.func @batch_matmul_secret_secret(%arg0: !secret.secret<tensor<2x17x19xf32>> {tensor_ext.layout = #layout1}, %arg1: !secret.secret<tensor<2x19x21xf32>> {tensor_ext.layout = #layout2}) -> (!secret.secret<tensor<2x17x21xf32>> {tensor_ext.layout = #layout}) {
    %cst = arith.constant dense<0.000000e+00> : tensor<2x17x21xf32>
    %0 = secret.generic(%arg0: !secret.secret<tensor<2x17x19xf32>> {tensor_ext.layout = #layout1}, %arg1: !secret.secret<tensor<2x19x21xf32>> {tensor_ext.layout = #layout2}) {
    ^body(%input0: tensor<2x17x19xf32>, %input1: tensor<2x19x21xf32>):
      %1 = tensor_ext.assign_layout %cst {layout = #layout3, tensor_ext.layout = #layout3} : tensor<2x17x21xf32>
      %2 = linalg.batch_matmul {secret.kernel = #kernel, tensor_ext.layout = #layout} ins(%input0, %input1 : tensor<2x17x19xf32>, tensor<2x19x21xf32>) outs(%1 : tensor<2x17x21xf32>) -> tensor<2x17x21xf32>
      secret.yield %2 : tensor<2x17x21xf32>
    } -> (!secret.secret<tensor<2x17x21xf32>> {tensor_ext.layout = #layout})
    return %0 : !secret.secret<tensor<2x17x21xf32>>
  }
}
