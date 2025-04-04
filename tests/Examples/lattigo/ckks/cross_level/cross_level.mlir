module attributes {backend.lattigo, ckks.schemeParam = #ckks.scheme_param<logN = 14, Q = [36028797019389953, 35184372121601, 35184372744193, 35184373006337, 35184373989377], P = [36028797019488257, 36028797020209153], logDefaultScale = 45>, scheme.ckks} {
  func.func @cross_level(%base: tensor<4xf32> {secret.secret}, %add: tensor<4xf32> {secret.secret}) -> tensor<4xf32> {
    // same level
    %base0 = arith.addf %base, %add : tensor<4xf32>
    // increase one level
    %mul1 = arith.mulf %base0, %base0 : tensor<4xf32>
    // cross level add
    %base1 = arith.addf %mul1, %add : tensor<4xf32>
    // increase one level
    %mul2 = arith.mulf %base1, %base1 : tensor<4xf32>
    // cross level add
    %base2 = arith.addf %mul2, %add : tensor<4xf32>
    // increase one level
    %mul3 = arith.mulf %base2, %base2 : tensor<4xf32>
    // cross level add
    %base3 = arith.addf %mul3, %add : tensor<4xf32>
    return %base3 : tensor<4xf32>
  }
}
