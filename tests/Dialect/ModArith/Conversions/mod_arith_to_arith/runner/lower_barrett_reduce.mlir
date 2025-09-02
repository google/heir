func.func public @test_lower_barrett_reduce() -> memref<4xi32> {
  %coeffs = arith.constant dense<[29498763, 58997760, 17, 7681]> : tensor<4xi26>
  %1 = mod_arith.barrett_reduce %coeffs { modulus = 7681 } : tensor<4xi26>

  %2 = arith.extui %1 : tensor<4xi26> to tensor<4xi32>
  %3 = bufferization.to_buffer %2 : tensor<4xi32> to memref<4xi32>
  return %3 : memref<4xi32>
}
