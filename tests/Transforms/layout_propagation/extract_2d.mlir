// RUN: heir-opt %s --layout-propagation=ciphertext-size=16 | FileCheck %s

// CHECK: [[input_layout:#.*]] = #tensor_ext.layout<"{ [i0, i1] -> [ct, slot] : i0 = 0 and ct = 0 and (-i1 + slot) mod 16 = 0 and 0 <= i1 <= 15 and 0 <= slot <= 15 }">
// CHECK: [[scalar_layout_extract:#.*]] = #tensor_ext.layout<"{ [] -> [ct, slot] : ct = 0 and slot = 0 }">
// CHECK: [[scalar_layout_default:#.*]] = #tensor_ext.layout<"{ [] -> [ct, slot] : ct = 0 and 0 <= slot <= 15 }">

// CHECK: secret.generic
// CHECK-NEXT: body
// CHECK-NEXT: tensor.extract
// CHECK-SAME: [[scalar_layout_extract]]
// CHECK-NEXT: tensor.extract
// CHECK-SAME: [[scalar_layout_extract]]
// CHECK-NEXT: tensor_ext.assign_layout
// CHECK-SAME: [[scalar_layout_default]]
// CHECK-NEXT: tensor_ext.convert_layout
// CHECK-NEXT: arith.mulf
// CHECK-NEXT: arith.addf
// CHECK-NEXT: tensor_ext.convert_layout
// CHECK-NEXT: tensor.insert
module {
  func.func @main(%arg0: !secret.secret<tensor<1x16xf32>>, %arg1: !secret.secret<tensor<1x16xf32>>) -> !secret.secret<tensor<1x16xf32>> {
    %cst_254 = arith.constant 0.0992246866 : f32
    %c0 = arith.constant 0 : index
    %0 = secret.generic(%arg0: !secret.secret<tensor<1x16xf32>>, %arg1: !secret.secret<tensor<1x16xf32>>) {
    ^body(%input0: tensor<1x16xf32>, %input1: tensor<1x16xf32>):
      %extracted = tensor.extract %input0[%c0, %c0] : tensor<1x16xf32>
      %extracted_255 = tensor.extract %input1[%c0, %c0] : tensor<1x16xf32>
      %1 = arith.mulf %extracted, %cst_254 : f32
      %2 = arith.addf %extracted_255, %1 : f32
      %inserted = tensor.insert %2 into %input1[%c0, %c0] : tensor<1x16xf32>
      secret.yield %inserted : tensor<1x16xf32>
    } -> !secret.secret<tensor<1x16xf32>>
    return %0 : !secret.secret<tensor<1x16xf32>>
  }
}
