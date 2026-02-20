
module {
  func.func @matvec(%arg0 : tensor<784xf32> {secret.secret}) -> tensor<512xf32> {
    %matrix = arith.constant dense<1.0> : tensor<512x784xf32>
    %out = arith.constant dense<0.0> : tensor<512xf32>
    %0 = linalg.matvec ins(%matrix, %arg0 : tensor<512x784xf32>, tensor<784xf32>) outs(%out : tensor<512xf32>) -> tensor<512xf32>
    return %0 : tensor<512xf32>
  }
}
