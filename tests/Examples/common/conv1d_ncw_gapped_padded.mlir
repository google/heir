// A two-layer 1-D conv chain. The first conv has stride 2, so it leaves its
// result in a gapped packing. The second conv pads that result, so its data
// operand is both gapped and padded. LayoutPropagation must fold the pad into
// the second conv's own padding parameter and absorb the gapped packing into
// its plaintext diagonal filter, with no online layout conversion.
module {
  func.func @conv1d_ncw_gapped_padded(%arg0 : tensor<1x2x8xf32> {secret.secret}) -> tensor<1x2x4xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %out1 = arith.constant dense<0.000000e+00> : tensor<1x2x4xf32>
    %out2 = arith.constant dense<0.000000e+00> : tensor<1x2x4xf32>
    %filter1 = arith.constant dense<[[[-1.0, -0.5, 0.0], [0.5, 1.0, -1.0]], [[-0.5, 0.0, 0.5], [1.0, -1.0, -0.5]]]> : tensor<2x2x3xf32>
    %filter2 = arith.constant dense<[[[0.5, -1.0, 0.25], [-0.25, 0.5, 1.0]], [[1.0, 0.25, -0.5], [0.0, -1.0, 0.5]]]> : tensor<2x2x3xf32>
    %padded1 = tensor.pad %arg0 low[0, 0, 1] high[0, 0, 1] {
    ^bb0(%a: index, %b: index, %c: index):
      tensor.yield %cst : f32
    } : tensor<1x2x8xf32> to tensor<1x2x10xf32>
    %conv1 = linalg.conv_1d_ncw_fcw {dilations = dense<1> : vector<1xi64>, strides = dense<2> : vector<1xi64>} ins(%padded1, %filter1 : tensor<1x2x10xf32>, tensor<2x2x3xf32>) outs(%out1 : tensor<1x2x4xf32>) -> tensor<1x2x4xf32>
    %padded2 = tensor.pad %conv1 low[0, 0, 1] high[0, 0, 1] {
    ^bb0(%a: index, %b: index, %c: index):
      tensor.yield %cst : f32
    } : tensor<1x2x4xf32> to tensor<1x2x6xf32>
    %conv2 = linalg.conv_1d_ncw_fcw {dilations = dense<1> : vector<1xi64>, strides = dense<1> : vector<1xi64>} ins(%padded2, %filter2 : tensor<1x2x6xf32>, tensor<2x2x3xf32>) outs(%out2 : tensor<1x2x4xf32>) -> tensor<1x2x4xf32>
    return %conv2 : tensor<1x2x4xf32>
  }
}
