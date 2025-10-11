// RUN: heir-opt %s --convert-to-ciphertext-semantics=ciphertext-size=1024 | FileCheck %s

!data_ty = !secret.secret<tensor<8xi16>>
#tensor_layout = #tensor_ext.layout<"{ [i0] -> [ct, slot] : (slot - i0) mod 8 = 0 and ct = 0 and 1023 >= slot >= 0 and 7 >= i0 >= 0 }">

#scalar_layout = #tensor_ext.layout<"{ [] -> [ct, slot] : ct = 0 and 0 <= slot <= 1023 }">

// CHECK: dot_product
// CHECK: secret.secret<tensor<1x1024xi16>>
func.func @dot_product(%arg0: !data_ty {tensor_ext.layout = #tensor_layout}, %arg1: !data_ty {tensor_ext.layout = #tensor_layout}) -> (!secret.secret<i16> {tensor_ext.layout = #scalar_layout}) {
  %c0_i16 = arith.constant 0 : i16
  %c0_laidout = tensor_ext.assign_layout %c0_i16 {layout = #scalar_layout, tensor_ext.layout = #scalar_layout} : i16
  %0 = secret.generic(%arg0: !secret.secret<tensor<8xi16>> {tensor_ext.layout = #tensor_layout}, %arg1: !secret.secret<tensor<8xi16>> {tensor_ext.layout = #tensor_layout}) {
  ^body(%input0: tensor<8xi16>, %input1: tensor<8xi16>):
    %1 = affine.for %arg2 = 0 to 8 iter_args(%arg3 = %c0_laidout) -> (i16) {
      %extracted = tensor.extract %input0[%arg2] {tensor_ext.layout = #scalar_layout} : tensor<8xi16>
      %extracted_0 = tensor.extract %input1[%arg2] {tensor_ext.layout = #scalar_layout} : tensor<8xi16>
      %2 = arith.muli %extracted, %extracted_0 {tensor_ext.layout = #scalar_layout} : i16
      %3 = arith.addi %arg3, %2 {tensor_ext.layout = #scalar_layout} : i16
      affine.yield %3 : i16
    } {__argattrs = [{tensor_ext.layout = #scalar_layout}], tensor_ext.layout = #scalar_layout}
    secret.yield %1 {tensor_ext.layout = []} : i16
  } -> (!secret.secret<i16> {tensor_ext.layout = #scalar_layout})
  return %0 : !secret.secret<i16>
}
