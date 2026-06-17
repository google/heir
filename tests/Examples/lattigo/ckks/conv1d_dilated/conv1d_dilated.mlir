module {
  func.func @conv1d_dilated(%arg0: tensor<1x1x28xf32> {secret.secret}) -> tensor<1x4x24xf32> {
    %cst_0 = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<1x4x24xf32>
    %1 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<1x4x24xf32>) -> tensor<1x4x24xf32>
    %filter = arith.constant dense<1.000000e+00> : tensor<4x1x3xf32>
    %2 = linalg.conv_1d_ncw_fcw {dilations = dense<2> : vector<1xi64>, strides = dense<1> : vector<1xi64>} ins(%arg0, %filter : tensor<1x1x28xf32>, tensor<4x1x3xf32>) outs(%1 : tensor<1x4x24xf32>) -> tensor<1x4x24xf32>
    return %2 : tensor<1x4x24xf32>
  }
}
