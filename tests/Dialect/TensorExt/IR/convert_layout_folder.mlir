// RUN: heir-opt --canonicalize --split-input-file %s | FileCheck %s

#layout1 = #tensor_ext.new_layout<"{ [row, col] -> [ct, slot] : (slot - row) mod 16 = 0 and (ct + slot - col) mod 16 = 0 and row >= 0 and col >= 0 and ct >= 0 and slot >= 0 and 31 >= slot and 15 >= ct and 15 >= row and 15 >= col }">
#layout2 = #tensor_ext.new_layout<"{ [row, col] -> [ct, slot] : (slot - row) mod 16 = 1 and (ct + slot - col) mod 16 = 5 and row >= 0 and col >= 0 and ct >= 0 and slot >= 0 and 31 >= slot and 15 >= ct and 15 >= row and 15 >= col }">

// CHECK: func @fold
// CHECK-NOT: tensor_ext.convert_layout
// CHECK: return
func.func @fold(%arg0: !secret.secret<tensor<1x32xi16>> {tensor_ext.layout = #layout1}) -> (!secret.secret<tensor<1x32xi16>> {tensor_ext.layout = #layout1}) {
  %0 = secret.generic(%arg0 : !secret.secret<tensor<1x32xi16>>) attrs = {arg0 = {layout = #layout1}, layout = [#layout1]} {
  ^body(%input0: tensor<1x32xi16>):
    %1 = tensor_ext.convert_layout %input0 {from_layout = #layout1, tensor_ext.layout = [#layout2], to_layout = #layout2} : tensor<1x32xi16>
    %2 = tensor_ext.convert_layout %1 {from_layout = #layout2, layout = [#layout1], to_layout = #layout1} : tensor<1x32xi16>
    secret.yield %2 : tensor<1x32xi16>
  } -> !secret.secret<tensor<1x32xi16>>
  return %0 : !secret.secret<tensor<1x32xi16>>
}

// -----

#layout1 = #tensor_ext.new_layout<"{ [row, col] -> [ct, slot] : (slot - row) mod 16 = 0 and (ct + slot - col) mod 16 = 0 and row >= 0 and col >= 0 and ct >= 0 and slot >= 0 and 31 >= slot and 15 >= ct and 15 >= row and 15 >= col }">

// CHECK: func @noop
// CHECK-NOT: tensor_ext.convert_layout
// CHECK: return
func.func @noop(%arg0: !secret.secret<tensor<1x32xi16>> {tensor_ext.layout = #layout1}) -> (!secret.secret<tensor<1x32xi16>> {tensor_ext.layout = #layout1}) {
  %0 = secret.generic(%arg0 : !secret.secret<tensor<1x32xi16>>) attrs = {arg0 = {layout = #layout1}, layout = [#layout1]} {
  ^body(%input0: tensor<1x32xi16>):
    %1 = tensor_ext.convert_layout %input0 {from_layout = #layout1, layout = [#layout1], to_layout = #layout1} : tensor<1x32xi16>
    secret.yield %1 : tensor<1x32xi16>
  } -> !secret.secret<tensor<1x32xi16>>
  return %0 : !secret.secret<tensor<1x32xi16>>
}
