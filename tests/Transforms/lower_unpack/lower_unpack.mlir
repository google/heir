// RUN: heir-opt --lower-unpack %s | FileCheck %s

#scalar_layout = #tensor_ext.layout<"{ [] -> [ct, slot] : ct = 0 and 0 <= slot <= 1023 }">
#scalar_original_type = #tensor_ext.original_type<originalType = i16, layout = #scalar_layout>

// CHECK: @unpack_scalar
// CHECK: arith.constant 0
// CHECK: tensor.extract
func.func @unpack_scalar(%arg0: tensor<1x1024xi16> {tensor_ext.original_type = #scalar_original_type}) -> i16 {
  %0 = tensor_ext.unpack %arg0 {layout=#scalar_layout} : (tensor<1x1024xi16>) -> i16
  return %0 : i16
}

#tensor_layout = #tensor_ext.layout<"{ [i0] -> [ct, slot] : (i0 - slot) mod 1024 = 3 and 31 >= i0 >= 0 and 1023 >= slot >= 0 and ct = 0 }">
#tensor_original_type = #tensor_ext.original_type<originalType = tensor<32xi16>, layout = #tensor_layout>

// CHECK: @unpack_rotated_tensor
// CHECK: arith.constant dense<0> : tensor<32xi16>
// CHECK: scf.for
// CHECK: scf.if
// CHECK: tensor.extract
// CHECK: tensor.insert
// CHECK: scf.yield
// CHECK: return
func.func @unpack_rotated_tensor(%arg0: tensor<1x1024xi16> {tensor_ext.original_type = #tensor_original_type}) -> tensor<32xi16> {
  %0 = tensor_ext.unpack %arg0 {layout=#tensor_layout} : (tensor<1x1024xi16>) -> tensor<32xi16>
  return %0 : tensor<32xi16>
}

// size 32 tensor repeated 64 size-64 ciphertexts
#tensor_layout2 = #tensor_ext.layout<"{ [col] -> [ct, slot] : 0 <= ct <= 64 and (slot - col) mod 64 = 0 and 0 <= col <= 63 and 0 <= slot <= 63 }">
#tensor_original_type2 = #tensor_ext.original_type<originalType = tensor<32xi16>, layout = #tensor_layout2>

// CHECK: @unpack_rotated_tensor
// CHECK: scf.for
// CHECK: scf.for
// CHECK: return
func.func @unpack_rotated_tensor2(%arg0: tensor<64x64xi16> {tensor_ext.original_type = #tensor_original_type2}) -> tensor<32xi16> {
  %0 = tensor_ext.unpack %arg0 {layout=#tensor_layout2} : (tensor<64x64xi16>) -> tensor<32xi16>
  return %0 : tensor<32xi16>
}
