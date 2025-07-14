// RUN: heir-opt %s --convert-to-ciphertext-semantics=ciphertext-size=1024 | FileCheck %s

#alignment = #tensor_ext.alignment<in = [8], out = [8]>
#layout = #tensor_ext.layout<map = (d0) -> (d0 mod 1024), alignment = #alignment>

#scalar_alignment = #tensor_ext.alignment<in = [], out = [1024], insertedDims = [0]>
#scalar_layout = #tensor_ext.layout<map = (d0) -> (d0 mod 1024), alignment = #scalar_alignment>

// CHECK: [[alignment:[^ ]*]] = #tensor_ext.alignment<in = [], out = [1024], insertedDims = [0]>
// CHECK: [[map:[^ ]*]] = affine_map<(d0) -> (d0 mod 1024)>
// CHECK: [[layout:[^ ]*]] = #tensor_ext.layout<map = (d0) -> (d0 mod 1024), alignment = [[alignment]]>
// CHECK: [[original_type:[^ ]*]] = #tensor_ext.original_type<originalType = i16, layout = [[layout]]>
// CHECK: module {
// CHECK:   func.func @extract([[arg0:[^:]*]]: !secret.secret<tensor<1024xi16>>
// CHECK-SAME:    -> (!secret.secret<tensor<1024xi16>> {tensor_ext.original_type = [[original_type]]}) {
// CHECK:     [[c3:[^ ]*]] = arith.constant 3
// CHECK:     [[v1:[^ ]*]] = secret.generic([[arg0]]: !secret.secret<tensor<1024xi16>>) {
// CHECK:     ^body([[input0:[^:]*]]: tensor<1024xi16>):
// CHECK:       [[v2:%[^ ]*]] = affine.apply [[map]]([[c3]])
// CHECK-DAG:   [[cst_0:[^ ]*]] = arith.constant dense<0>
// CHECK-DAG:       [[c1_i16:[^ ]*]] = arith.constant 1
// CHECK:       [[inserted:[^ ]*]] = tensor.insert [[c1_i16]] into [[cst_0]][
// CHECK-SAME:        [[v2]]
// CHECK:       [[v3:[^ ]*]] = arith.muli [[inserted]], [[input0]]
// CHECK:       [[v4:[^ ]*]] = tensor_ext.rotate [[v3]], [[v2]]
// CHECK:       secret.yield [[v4]]
//
// CHECK:     return [[v1]]
// CHECK:   }
// CHECK: }

func.func @extract(%arg0: !secret.secret<tensor<8xi16>> {tensor_ext.layout = #layout}) -> (!secret.secret<i16> {tensor_ext.layout = #scalar_layout}) {
  %index = arith.constant 3 : index
  %0 = secret.generic(%arg0 : !secret.secret<tensor<8xi16>> {tensor_ext.layout = #layout}) {
  ^body(%input0: tensor<8xi16>):
    %0 = tensor.extract %input0[%index] {tensor_ext.layout = #scalar_layout} : tensor<8xi16>
    secret.yield %0 : i16
  } -> (!secret.secret<i16> {tensor_ext.layout = #layout})
  return %0 : !secret.secret<i16>
}
