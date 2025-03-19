// RUN: heir-opt %s --convert-to-ciphertext-semantics=ciphertext-size=1024 | FileCheck %s

#alignment = #tensor_ext.alignment<in = [], out = [1], insertedDims = [0]>
#alignment1 = #tensor_ext.alignment<in = [8], out = [8]>
#layout = #tensor_ext.layout<map = (d0) -> (d0 mod 1024), alignment = #alignment>
#layout1 = #tensor_ext.layout<map = (d0) -> (d0 mod 1024), alignment = #alignment1>
// CHECK: dot_product
func.func @dot_product(%arg0: !secret.secret<tensor<8xi16>> {tensor_ext.layout = #layout1}, %arg1: !secret.secret<tensor<8xi16>> {tensor_ext.layout = #layout1}) -> (!secret.secret<i16> {tensor_ext.layout = #layout}) {
  %c0_i16 = arith.constant 0 : i16
  %c0_laidout = tensor_ext.assign_layout %c0_i16 {layout = #layout, tensor_ext.layout = #layout} : i16
  %0 = secret.generic ins(%arg0, %arg1 : !secret.secret<tensor<8xi16>>, !secret.secret<tensor<8xi16>>) attrs = {__argattrs = [{tensor_ext.layout = #layout1}, {tensor_ext.layout = #layout1}], __resattrs = [{tensor_ext.layout = #layout}]} {
  ^body(%input0: tensor<8xi16>, %input1: tensor<8xi16>):
    %1 = affine.for %arg2 = 0 to 8 iter_args(%arg3 = %c0_laidout) -> (i16) {
      %extracted = tensor.extract %input0[%arg2] {tensor_ext.layout = #layout} : tensor<8xi16>
      %extracted_0 = tensor.extract %input1[%arg2] {tensor_ext.layout = #layout} : tensor<8xi16>
      %2 = arith.muli %extracted, %extracted_0 {tensor_ext.layout = #layout} : i16
      %3 = arith.addi %arg3, %2 {tensor_ext.layout = #layout} : i16
      affine.yield %3 : i16
    } {__argattrs = [{tensor_ext.layout = #layout}], tensor_ext.layout = #layout}
    secret.yield %1 {tensor_ext.layout = []} : i16
  } -> !secret.secret<i16>
  return %0 : !secret.secret<i16>
}
