// A stride-2 conv whose 3 output channels are not a multiple of the gap 2. The
// shuffled layout reserves a whole pair of channels, so one channel stays empty
// and the Toeplitz matrix gets zero rows for it.
module {
  func.func @conv1d_channel_pad(%arg0: tensor<1x2x8xf32> {secret.secret}) -> tensor<1x3x4xf32> {
    %filter = arith.constant dense<[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]], [[9.0, 10.0], [11.0, 12.0]]]> : tensor<3x2x2xf32>
    %out = arith.constant dense<0.0> : tensor<1x3x4xf32>
    %0 = linalg.conv_1d_ncw_fcw {dilations = dense<1> : vector<1xi64>, strides = dense<2> : vector<1xi64>} ins(%arg0, %filter : tensor<1x2x8xf32>, tensor<3x2x2xf32>) outs(%out : tensor<1x3x4xf32>) -> tensor<1x3x4xf32>
    return %0 : tensor<1x3x4xf32>
  }
}
