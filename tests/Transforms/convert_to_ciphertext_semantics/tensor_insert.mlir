// RUN: heir-opt %s --convert-to-ciphertext-semantics=ciphertext-size=1024 | FileCheck %s

#alignment = #tensor_ext.alignment<in = [8], out = [8]>
#layout = #tensor_ext.layout<map = (d0) -> (d0 mod 1024), alignment = #alignment>

#scalar_alignment = #tensor_ext.alignment<in = [], out = [1], insertedDims = [0]>
#scalar_layout = #tensor_ext.layout<map = (d0) -> (d0 mod 1024), alignment = #scalar_alignment>

// CHECK: [[alignment:[^ ]*]] = #tensor_ext.alignment<in = [8], out = [8]>
// CHECK: [[map:[^ ]*]] = affine_map<(d0) -> (d0)>
// CHECK: [[map1:[^ ]*]] = affine_map<(d0) -> (d0 mod 1024)>
// CHECK: [[layout:[^ ]*]] = #tensor_ext.layout<map = (d0) -> (d0 mod 1024), alignment = [[alignment]]>
// CHECK: [[original_type:[^ ]*]] = #tensor_ext.original_type<originalType = tensor<8xi16>, layout = [[layout]]>
// CHECK: module {
// CHECK:   func.func @insert([[arg0:[^:]*]]: !secret.secret<tensor<1024xi16>> {tensor_ext.original_type = [[original_type]]}) -> (!secret.secret<tensor<1024xi16>> {tensor_ext.original_type = [[original_type]]}) {
// CHECK:     [[c3:[^ ]*]] = arith.constant 3
// CHECK:     [[c7_i16:[^ ]*]] = arith.constant 7
// CHECK:     [[splat:[^ ]*]] = tensor.splat %c7_i16 : tensor<1xi16>
// CHECK:     [[cst:[^ ]*]] = arith.constant dense<0> : tensor<1024xi16>
// CHECK:     [[v0:[^ ]*]] = linalg.generic
// CHECK-SAME:   ins([[splat]] : tensor<1xi16>) outs([[cst]] : tensor<1024xi16>) {
//
// CHECK:     [[v1:[^ ]*]] = secret.generic ins([[arg0]] : !secret.secret<tensor<1024xi16>>) {
// CHECK:     ^body([[input0:[^:]*]]: tensor<1024xi16>):
// CHECK:       [[v2:[^ ]*]] = affine.apply #map1([[c3]])
// CHECK:       [[c0:[^ ]*]] = arith.constant 0
// CHECK:       [[cst_0:[^ ]*]] = arith.constant dense<0>
// CHECK:       [[c1_i16:[^ ]*]] = arith.constant 1
// CHECK:       [[inserted:[^ ]*]] = tensor.insert [[c1_i16]] into [[cst_0]][
// CHECK-SAME:        [[c0]]
// CHECK:       [[v3:[^ ]*]] = arith.muli [[inserted]], [[v0]]
// CHECK:       [[v4:[^ ]*]] = tensor_ext.rotate [[v3]], [[v2]]
// CHECK:       [[cst_1:[^ ]*]] = arith.constant dense<1>
// CHECK:       [[c0_i16:[^ ]*]] = arith.constant 0
// CHECK:       [[inserted_2:[^ ]*]] = tensor.insert [[c0_i16]] into [[cst_1]]
// CHECK-SAME:        [[v2]]
// CHECK:       [[v5:[^ ]*]] = arith.muli [[inserted_2]], [[input0]]
// CHECK:       [[v6:[^ ]*]] = arith.addi [[v3]], [[v5]]
// CHECK:       secret.yield [[v6]]
//
// CHECK:     return [[v1]]
// CHECK:   }
// CHECK: }

func.func @insert(%arg0: !secret.secret<tensor<8xi16>> {tensor_ext.layout = #layout}) -> (!secret.secret<tensor<8xi16>> {tensor_ext.layout = #layout}) {
  %insertIndex = arith.constant 3 : index
  %c7 = arith.constant 7 : i16
  %c7_laidout = tensor_ext.assign_layout %c7 {layout = #scalar_layout, tensor_ext.layout = #scalar_layout} : i16
  %0 = secret.generic ins(%arg0 : !secret.secret<tensor<8xi16>>) attrs = {__argattrs = [{tensor_ext.layout = #layout}], __resattrs = [{tensor_ext.layout = #layout}]} {
  ^body(%input0: tensor<8xi16>):
    %0 = tensor.insert %c7_laidout into %input0[%insertIndex] {tensor_ext.layout = #layout} : tensor<8xi16>
    secret.yield %0 : tensor<8xi16>
  } -> !secret.secret<tensor<8xi16>>
  return %0 : !secret.secret<tensor<8xi16>>
}
