func.func @simple_sum(%arg0: tensor<32xf32> {secret.secret}) -> f32 {
  %c0 = arith.constant 0 : index
  %c0_f32 = arith.constant 0.0 : f32
  %0 = affine.for %i = 0 to 32 iter_args(%sum_iter = %c0_f32) -> f32 {
    %1 = tensor.extract %arg0[%i] : tensor<32xf32>
    %2 = arith.addf %1, %sum_iter : f32
    affine.yield %2 : f32
  }
  return %0 : f32
}
