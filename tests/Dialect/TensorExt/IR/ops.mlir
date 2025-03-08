// RUN: heir-opt %s

// Test for syntax

func.func @test_rotate(%0: tensor<16xi32>) -> tensor<16xi32> {
  %c1 = arith.constant 1 : i32
  %1 = tensor_ext.rotate %0, %c1 : tensor<16xi32>, i32
  return %1 : tensor<16xi32>
}

#row_major = affine_map<(d0, d1) -> (16*d0 + d1)>
#col_major = affine_map<(d0, d1) -> (16*d1 + d0)>
func.func @test_convert_layout(%0: tensor<16x16xi32>) -> tensor<16x16xi32> {
  %1 = tensor_ext.convert_layout %0 {from_layout = #row_major, to_layout = #col_major} : tensor<16x16xi32>
  return %1 : tensor<16x16xi32>
}

func.func @test_assign_layout(%0: tensor<16x16xi32>) -> tensor<16x16xi32> {
  %1 = tensor_ext.assign_layout %0 {layout = #row_major} : tensor<16x16xi32>
  return %1 : tensor<16x16xi32>
}
