// RUN: heir-opt %s --convert-to-ciphertext-semantics="ciphertext-size=1024 unroll-kernels=false" | FileCheck %s
//
// Ensure that the if guards are inserted properly for a non-square matvec kernel
//
// CHECK:      @matvec
// CHECK:      scf.for
// CHECK:      scf.for
// CHECK:      scf.if
// CHECK-NEXT:   tensor.extract_slice
// CHECK-NEXT:   arith.muli
// CHECK-NEXT:   arith.muli
// CHECK-NEXT:   tensor_ext.rotate
// CHECK-NEXT:   tensor_ext.rotate
// CHECK-NEXT:   arith.mulf
// CHECK-NEXT:   arith.addf
// CHECK-NEXT:   scf.yield
// CHECK-NEXT: } else {
// CHECK-NEXT:   scf.yield
// CHECK-NEXT: }


#kernel = #secret.kernel<name = "MatvecDiagonal", force = false>
#layout = #tensor_ext.layout<"{ [i0] -> [ct, slot] : ct = 0 and (-i0 + slot) mod 512 = 0 and 0 <= i0 <= 511 and 0 <= slot <= 1023 }">
#layout1 = #tensor_ext.layout<"{ [i0] -> [ct, slot] : ct = 0 and (-i0 + slot) mod 1024 = 0 and 0 <= i0 <= 783 and 0 <= slot <= 1023 }">
#layout2 = #tensor_ext.layout<"{ [i0, i1] -> [ct, slot] : (i0 - i1 + ct) mod 512 = 0 and (-i1 + ct + slot) mod 1024 = 0 and 0 <= i0 <= 511 and 0 <= i1 <= 783 and 0 <= ct <= 511 and 0 <= slot <= 1023 }">
module attributes {backend.lattigo, scheme.ckks} {
  func.func @matvec(%arg0: !secret.secret<tensor<784xf32>> {tensor_ext.layout = #layout1}) -> (!secret.secret<tensor<512xf32>> {tensor_ext.layout = #layout}) {
    %cst = arith.constant dense<0.000000e+00> : tensor<512xf32>
    %cst_0 = arith.constant dense<1.000000e+00> : tensor<512x784xf32>
    %0 = secret.generic(%arg0: !secret.secret<tensor<784xf32>> {tensor_ext.layout = #layout1}) {
    ^body(%input0: tensor<784xf32>):
      %1 = tensor_ext.assign_layout %cst_0 {layout = #layout2, tensor_ext.layout = #layout2} : tensor<512x784xf32>
      %2 = tensor_ext.assign_layout %cst {layout = #layout, tensor_ext.layout = #layout} : tensor<512xf32>
      %3 = linalg.matvec {secret.kernel = #kernel, tensor_ext.layout = #layout} ins(%1, %input0 : tensor<512x784xf32>, tensor<784xf32>) outs(%2 : tensor<512xf32>) -> tensor<512xf32>
      secret.yield %3 : tensor<512xf32>
    } -> (!secret.secret<tensor<512xf32>> {tensor_ext.layout = #layout})
    return %0 : !secret.secret<tensor<512xf32>>
  }
}
