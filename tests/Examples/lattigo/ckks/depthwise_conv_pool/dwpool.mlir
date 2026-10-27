// Zero-padded depthwise 3x3 convolution (generalized), a sum pool as a
// linalg.generic, and a linear layer whose weights fold in the 1/36 average.

#input = affine_map<(n, c, h, w) -> (n, c, h, w)>
#pooled = affine_map<(n, c, h, w) -> (n, c)>

module {
  func.func @dwpool(%arg0: tensor<1x2x6x6xf32> {secret.secret}) -> tensor<1x3xf32> {
    %zero = arith.constant 0.000000e+00 : f32
    %padded = tensor.pad %arg0 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%i0: index, %i1: index, %i2: index, %i3: index):
      tensor.yield %zero : f32
    } : tensor<1x2x6x6xf32> to tensor<1x2x8x8xf32>
    %filter = arith.constant dense<[[[0.1, 0.2, 0.1], [0.2, 0.4, 0.2], [0.1, 0.2, 0.1]], [[0.0, -0.25, 0.0], [-0.25, 1.0, -0.25], [0.0, -0.25, 0.0]]]> : tensor<2x3x3xf32>
    %empty = tensor.empty() : tensor<1x2x6x6xf32>
    %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<1x2x6x6xf32>) -> tensor<1x2x6x6xf32>
    %conv = linalg.depthwise_conv_2d_nchw_chw {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%padded, %filter : tensor<1x2x8x8xf32>, tensor<2x3x3xf32>) outs(%init : tensor<1x2x6x6xf32>) -> tensor<1x2x6x6xf32>
    %pool_empty = tensor.empty() : tensor<1x2xf32>
    %pool_init = linalg.fill ins(%zero : f32) outs(%pool_empty : tensor<1x2xf32>) -> tensor<1x2xf32>
    %pool = linalg.generic {indexing_maps = [#input, #pooled], iterator_types = ["parallel", "parallel", "reduction", "reduction"]} ins(%conv : tensor<1x2x6x6xf32>) outs(%pool_init : tensor<1x2xf32>) {
    ^bb0(%in: f32, %acc: f32):
      %sum = arith.addf %acc, %in : f32
      linalg.yield %sum : f32
    } -> tensor<1x2xf32>
    %weights = arith.constant dense<[[0.027777778, 0.013888889, -0.027777778], [0.013888889, -0.027777778, 0.027777778]]> : tensor<2x3xf32>
    %out_empty = tensor.empty() : tensor<1x3xf32>
    %out_init = linalg.fill ins(%zero : f32) outs(%out_empty : tensor<1x3xf32>) -> tensor<1x3xf32>
    %out = linalg.matmul ins(%pool, %weights : tensor<1x2xf32>, tensor<2x3xf32>) outs(%out_init : tensor<1x3xf32>) -> tensor<1x3xf32>
    return %out : tensor<1x3xf32>
  }
}
