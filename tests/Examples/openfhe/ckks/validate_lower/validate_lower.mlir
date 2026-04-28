func.func @test_validate_lower(%arg0: tensor<8xf32> {secret.secret}) -> tensor<8xf32> {
  debug.validate %arg0 {name = "input_val", metadata = "input_meta"} : tensor<8xf32>
  %0 = arith.addf %arg0, %arg0 : tensor<8xf32>
  debug.validate %0 {name = "output_val", metadata = "output_meta"} : tensor<8xf32>
  return %0 : tensor<8xf32>
}
