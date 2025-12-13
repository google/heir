#map = affine_map<(d0, d1) -> (0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @fn_under_test(%arg0: tensor<1x1xi8> {secret.secret}) -> tensor<1x3xi16> {
    %cst = arith.constant dense<[[9, 54, 57]]> : tensor<1x3xi16>
    %cst_0 = tensor.empty() : tensor<1x3xi16>
    %c0_i16 = arith.constant 0 : i16
    %1 = linalg.quantized_matmul ins(%arg0, %cst, %c0_i16, %c0_i16 : tensor<1x1xi8>, tensor<1x3xi16>, i16, i16) outs(%cst_0 : tensor<1x3xi16>) -> tensor<1x3xi16>
    %bias = arith.constant dense<[[1, 2, 5438]]> : tensor<1x3xi16>
    %2 = arith.addi %1, %bias : tensor<1x3xi16>
    
    // CHECK: return
    return %2 : tensor<1x3xi16>
  }
}