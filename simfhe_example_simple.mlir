// Simple CKKS example for SimFHE demonstration
// This example shows polynomial evaluation: f(x,y) = x^2 + 2xy + y - 3

func.func @polynomial_evaluation(%x: tensor<16xf32> {secret.secret}, %y: tensor<16xf32> {secret.secret}) -> tensor<16xf32> {
  // Compute x^2
  %x_squared = arith.mulf %x, %x : tensor<16xf32>

  // Compute 2xy
  %xy = arith.mulf %x, %y : tensor<16xf32>
  %c2 = arith.constant dense<2.0> : tensor<16xf32>
  %two_xy = arith.mulf %xy, %c2 : tensor<16xf32>

  // Add x^2 + 2xy
  %sum1 = arith.addf %x_squared, %two_xy : tensor<16xf32>

  // Add y
  %sum2 = arith.addf %sum1, %y : tensor<16xf32>

  // Subtract 3
  %c3 = arith.constant dense<3.0> : tensor<16xf32>
  %result = arith.subf %sum2, %c3 : tensor<16xf32>

  return %result : tensor<16xf32>
}
