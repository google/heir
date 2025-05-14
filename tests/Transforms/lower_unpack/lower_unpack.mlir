// RUN: heir-opt --lower-unpack %s | FileCheck %s

// CHECK-DAG: [[map_id:.*]] = affine_map<(d0) -> (d0)>
// CHECK-DAG: [[map_shifted:.*]] = affine_map<(d0) -> ((d0 + 3) mod 1024)>

#scalar_alignment = #tensor_ext.alignment<in = [], out = [1024], insertedDims = [0]>
#scalar_layout = #tensor_ext.layout<map = (d0) -> (d0), alignment = #scalar_alignment>
#scalar_original_type = #tensor_ext.original_type<originalType = i16, layout = #scalar_layout>

// CHECK: @unpack_scalar
// CHECK: arith.constant 0
// CHECK: tensor.extract
func.func @unpack_scalar(%arg0: tensor<1024xi16> {tensor_ext.original_type = #scalar_original_type}) -> i16 {
  %0 = tensor_ext.unpack %arg0 {layout=#scalar_layout} : (tensor<1024xi16>) -> i16
  return %0 : i16
}

#tensor_alignment = #tensor_ext.alignment<in = [32], out = [1024]>
#tensor_layout = #tensor_ext.layout<map = (d0) -> ((d0 + 3) mod 1024), alignment = #tensor_alignment>
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

#tensor_alignment2 = #tensor_ext.alignment<in = [32], out = [64, 64], insertedDims=[0], padding=[63, 0], paddingValue=0>
#tensor_layout2 = #tensor_ext.layout<map = (d0, d1) -> (d0, d1), alignment = #tensor_alignment2>
#tensor_original_type2 = #tensor_ext.original_type<originalType = tensor<32xi16>, layout = #tensor_layout2>

// CHECK: @unpack_rotated_tensor
// CHECK-NEXT: tensor.extract_slice
// CHECK-SAME: [0, 0] [1, 32] [1, 1] : tensor<64x64xi16> to tensor<32xi16>
// CHECK: return
func.func @unpack_rotated_tensor2(%arg0: tensor<64x64xi16> {tensor_ext.original_type = #tensor_original_type2}) -> tensor<32xi16> {
  %0 = tensor_ext.unpack %arg0 {layout=#tensor_layout2} : (tensor<64x64xi16>) -> tensor<32xi16>
  return %0 : tensor<32xi16>
}
