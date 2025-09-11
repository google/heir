func.func public @test_lower_barrett_reduce() -> tensor<4xi26> {
  %coeffs = arith.constant dense<[29498763, 58997760, 17, 7681]> : tensor<4xi26>
  %1 = mod_arith.barrett_reduce %coeffs { modulus = 7681 } : tensor<4xi26>
  return %1 : tensor<4xi26>
}
