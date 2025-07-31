module {
  func.func @fn_under_test(%arg0: tensor<1x1xi8> {iree.identifier = "serving_default_dense_input:0", secret.secret, tf_saved_model.index_path = ["dense_input"]}) -> tensor<1x3xi16> {
    %cst = arith.constant dense<[[9, 54, 57]]> : tensor<1x3xi16>
    %cst_0 = arith.constant dense<[[1, 2, 5438]]> : tensor<1x3xi16>
    %c0_i16 = arith.constant 0 : i16
    %1 = linalg.quantized_matmul ins(%arg0, %cst, %c0_i16, %c0_i16 : tensor<1x1xi8>, tensor<1x3xi16>, i16, i16) outs(%cst_0 : tensor<1x3xi16>) -> tensor<1x3xi16>
    // CHECK: return
    return %1 : tensor<1x3xi16>
  }
}
