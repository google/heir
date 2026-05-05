module {
  func.func @matvec(%arg0 : tensor<4xf32> {secret.secret}) -> tensor<8xf32> {
    %matrix = arith.constant dense<[[0.0, 1.0, 2.0, 3.0],
                                    [4.0, 5.0, 6.0, 7.0],
                                    [8.0, 9.0, 10.0, 11.0],
                                    [12.0, 13.0, 14.0, 15.0],
                                    [16.0, 17.0, 18.0, 19.0],
                                    [20.0, 21.0, 22.0, 23.0],
                                    [24.0, 25.0, 26.0, 27.0],
                                    [28.0, 29.0, 30.0, 31.0]]> : tensor<8x4xf32>
    %out = arith.constant dense<0.0> : tensor<8xf32>
    %0 = linalg.matvec ins(%matrix, %arg0 : tensor<8x4xf32>, tensor<4xf32>) outs(%out : tensor<8xf32>) -> tensor<8xf32>
    return %0 : tensor<8xf32>
  }
}
