// A stride-2 conv whose 3 output channels are not a multiple of gap^2 = 4. The
// pixel-shuffled layout reserves a whole 2x2 channel block, so one channel of
// the block stays empty and the Toeplitz matrix gets zero rows for it.
module {
  func.func @conv2d_channel_pad(%arg0: tensor<1x1x4x4xf32> {secret.secret}) -> tensor<1x3x2x2xf32> {
    %filter = arith.constant dense<[[[[1.0, 2.0], [3.0, 4.0]]], [[[5.0, 6.0], [7.0, 8.0]]], [[[9.0, 10.0], [11.0, 12.0]]]]> : tensor<3x1x2x2xf32>
    %init = tensor.empty() : tensor<1x3x2x2xf32>
    %conv = linalg.conv_2d_nchw_fchw {strides = dense<2> : vector<2xi64>, dilations = dense<1> : vector<2xi64>}
      ins(%arg0, %filter : tensor<1x1x4x4xf32>, tensor<3x1x2x2xf32>)
      outs(%init : tensor<1x3x2x2xf32>) -> tensor<1x3x2x2xf32>
    return %conv : tensor<1x3x2x2xf32>
  }
}
