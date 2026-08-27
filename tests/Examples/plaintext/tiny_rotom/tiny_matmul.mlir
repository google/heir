// One secret 1x16 vector through a public 16x8 layer plus bias: the smallest
// shape that still spans several ciphertexts at a small slot count.
module @jit_func attributes {jax.uses_shape_polymorphism = false} {
  func.func public @tiny(%w: tensor<8x16xf32>, %b: tensor<8xf32>,
                         %x: tensor<1x16xf32> {secret.secret})
      -> (tensor<1x8xf32> {jax.result_info = "result[0]"}) {
    %0 = tensor.empty() : tensor<16x8xf32>
    %wt = linalg.transpose ins(%w : tensor<8x16xf32>) outs(%0 : tensor<16x8xf32>) permutation = [1, 0]
    %zero = arith.constant 0.000000e+00 : f32
    %1 = tensor.empty() : tensor<1x8xf32>
    %init = linalg.fill ins(%zero : f32) outs(%1 : tensor<1x8xf32>) -> tensor<1x8xf32>
    %mm = linalg.matmul ins(%x, %wt : tensor<1x16xf32>, tensor<16x8xf32>) outs(%init : tensor<1x8xf32>) -> tensor<1x8xf32>
    %2 = tensor.empty() : tensor<1x8xf32>
    %bb = linalg.broadcast ins(%b : tensor<8xf32>) outs(%2 : tensor<1x8xf32>) dimensions = [0]
    %3 = tensor.empty() : tensor<1x8xf32>
    %out = linalg.map { arith.addf } ins(%bb, %mm : tensor<1x8xf32>, tensor<1x8xf32>) outs(%3 : tensor<1x8xf32>)
    return %out : tensor<1x8xf32>
  }
}
