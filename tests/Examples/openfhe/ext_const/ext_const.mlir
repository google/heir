module attributes {scheme.bgv} {
  func.func @test_fn() -> tensor<4xi32> {
    %c = arith.constant dense<[1, 2, 3, 4]> : tensor<4xi32>
    return %c : tensor<4xi32>
  }
}
