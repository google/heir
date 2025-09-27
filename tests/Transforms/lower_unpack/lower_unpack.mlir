// RUN: heir-opt --lower-unpack %s | FileCheck %s

// CHECK-DAG: [[map_id:.*]] = affine_map<(d0) -> (d0)>
// CHECK-DAG: [[map_shifted:.*]] = affine_map<(d0) -> ((d0 + 3) mod 1024)>

#scalar_layout = #tensor_ext.new_layout<"{ [] -> [ct, slot] : ct = 0 and 0 <= slot <= 1023 }">
#scalar_original_type = #tensor_ext.original_type<originalType = i16, layout = #scalar_layout>

// CHECK: @unpack_scalar
// CHECK: arith.constant 0
// CHECK: tensor.extract
func.func @unpack_scalar(%arg0: tensor<1024xi16> {tensor_ext.original_type = #scalar_original_type}) -> i16 {
  %0 = tensor_ext.unpack %arg0 {layout=#scalar_layout} : (tensor<1024xi16>) -> i16
  return %0 : i16
}

#tensor_layout = #tensor_ext.new_layout<"{ [i0] -> [ct, slot] : (i0 - slot) mod 1024 = 3 and 31 >= i0 >= 0 and 1023 >= slot >= 0 and ct = 0 }">
#tensor_original_type = #tensor_ext.original_type<originalType = tensor<32xi16>, layout = #tensor_layout>

// CHECK: @unpack_rotated_tensor
// CHECK: arith.constant dense<0> : tensor<1024xi16>
// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = {{\[}}[[map_shifted]], [[map_id]]]
// CHECK: tensor.extract_slice
// CHECK-SAME: [0] [32] [1] : tensor<1024xi16> to tensor<32xi16>
// CHECK: return
func.func @unpack_rotated_tensor(%arg0: tensor<1024xi16> {tensor_ext.original_type = #tensor_original_type}) -> tensor<32xi16> {
  %0 = tensor_ext.unpack %arg0 {layout=#tensor_layout} : (tensor<1024xi16>) -> tensor<32xi16>
  return %0 : tensor<32xi16>
}

// size 32 tensor repeated 64 size-64 ciphertexts
#tensor_layout2 = #tensor_ext.new_layout<"{ [col] -> [ct, slot] : 0 <= ct <= 64 and (slot - col) mod 64 = 0 and 0 <= col <= 63 and 0 <= slot <= 63 }">
#tensor_original_type2 = #tensor_ext.original_type<originalType = tensor<32xi16>, layout = #tensor_layout2>

// CHECK: @unpack_rotated_tensor
// CHECK-NEXT: tensor.extract_slice
// CHECK-SAME: [0, 0] [1, 32] [1, 1] : tensor<64x64xi16> to tensor<32xi16>
// CHECK: return
func.func @unpack_rotated_tensor2(%arg0: tensor<64x64xi16> {tensor_ext.original_type = #tensor_original_type2}) -> tensor<32xi16> {
  %0 = tensor_ext.unpack %arg0 {layout=#tensor_layout2} : (tensor<64x64xi16>) -> tensor<32xi16>
  return %0 : tensor<32xi16>
}
