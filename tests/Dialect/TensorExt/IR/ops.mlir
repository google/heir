// RUN: heir-opt %s

// Test for syntax

func.func @test_rotate(%0: tensor<16xi32>) -> tensor<16xi32> {
  %c1 = arith.constant 1 : i32
  %1 = tensor_ext.rotate %0, %c1 : tensor<16xi32>, i32
  return %1 : tensor<16xi32>
}

#layout1 = #tensor_ext.layout<"{ [row, col] -> [ct, slot] : (slot - row) mod 16 = 0 and (ct + slot - col) mod 16 = 0 and row >= 0 and col >= 0 and ct >= 0 and slot >= 0 and 31 >= slot and 15 >= ct and 15 >= row and 15 >= col }">
#layout2 = #tensor_ext.layout<"{ [row, col] -> [ct, slot] : (slot - row) mod 16 = 1 and (ct + slot - col) mod 16 = 5 and row >= 0 and col >= 0 and ct >= 0 and slot >= 0 and 31 >= slot and 15 >= ct and 15 >= row and 15 >= col }">
func.func @test_convert_layout(%0: tensor<16x16xi32>) -> tensor<16x16xi32> {
  %1 = tensor_ext.convert_layout %0 {from_layout = #layout1, to_layout = #layout2} : tensor<16x16xi32>
  return %1 : tensor<16x16xi32>
}

func.func @test_assign_layout(%0: tensor<16x16xi32>) -> tensor<16x16xi32> {
  %1 = tensor_ext.assign_layout %0 {layout = #layout1} : tensor<16x16xi32>
  return %1 : tensor<16x16xi32>
}

func.func @test_rotate_and_reduce(%0: tensor<16xi32>) -> i32 {
  %c0 = arith.constant 0 : index
  %1 = arith.constant dense<1> : tensor<16xi32>
  %2 = tensor_ext.rotate_and_reduce %0, %1 {period = 1 : index, steps = 16 : index} : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
  %3 = tensor.extract %2[%c0] : tensor<16xi32>
  return %3 : i32
}

func.func @test_halevi_shoup_reduction(%0: tensor<16xi32>, %1: tensor<16x16xi32>) -> tensor<16xi32> {
  %2 = tensor_ext.rotate_and_reduce %0, %1 {period = 1 : index, steps = 16 : index} : (tensor<16xi32>, tensor<16x16xi32>) -> tensor<16xi32>
  return %2 : tensor<16xi32>
}
