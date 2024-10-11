// RUN: heir-opt %s

// Test for syntax

#enc = #tensor_ext.simd_packing<
      in = [7],
      padding = [1],
      padding_value = 0,
      out = [16]>

func.func @tensor_attributes(%0: tensor<16xi32, #enc>) -> tensor<16xi32, #enc> {
  return %0 : tensor<16xi32, #enc>
}

#enc0 = #tensor_ext.simd_packing<
      in = [7],
      padding = [1],
      out = [16]>

func.func @tensor_default_attributes(%0: tensor<16xi32, #enc0>) -> tensor<16xi32, #enc0> {
  return %0 : tensor<16xi32, #enc0>
}
