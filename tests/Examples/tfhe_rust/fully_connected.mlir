// This takes takes the input x and outputs 2 \cdot x + 1.
// #map = affine_map<(d0, d1) -> (0)>
// #map1 = affine_map<(d0, d1) -> (d0, d1)>
// module {
//   func.func @fn_under_test(%arg0: tensor<1x1xi8> {secret.secret}) -> tensor<1x1xi32> {
//     %cst = arith.constant dense<2> : tensor<1x1xi8>
//     %cst_0 = arith.constant dense<1> : tensor<1x1xi32>
//     %c0_i32 = arith.constant 0 : i32
//     %1 = linalg.quantized_matmul ins(%arg0, %cst, %c0_i32, %c0_i32 : tensor<1x1xi8>, tensor<1x1xi8>, i32, i32) outs(%cst_0 : tensor<1x1xi32>) -> tensor<1x1xi32>
//     return %1 : tensor<1x1xi32>
//   }
// }

module {
  func.func @main(%arg0: tensor<1x1xi8> {iree.identifier = "serving_default_dense_input:0", secret.secret, tf_saved_model.index_path = ["dense_input"]}) -> (tensor<1x2xi16> {iree.identifier = "StatefulPartitionedCall:0", tf_saved_model.index_path = ["dense_2"]}) attributes {tf_saved_model.exported_names = ["serving_default"]} {
    %cst = arith.constant dense<[[9, 54, 57]]> : tensor<1x3xi8>
    %cst_0 = arith.constant dense<0> : tensor<1x3xi16>
    %cst_1 = arith.constant dense<0> : tensor<1x2xi16>
    %c0_i16 = arith.constant 0 : i16
    %cst_3 = arith.constant dense<[[12, 9], [12, 26], [25, 36]]> : tensor<3x2xi8>
    %2 = linalg.quantized_matmul ins(%arg0, %cst, %c0_i16, %c0_i16 : tensor<1x1xi8>, tensor<1x3xi8>, i16, i16) outs(%cst_0 : tensor<1x3xi16>) -> tensor<1x3xi16>
    %4 = linalg.quantized_matmul ins(%2, %cst_3, %c0_i16, %c0_i16 : tensor<1x3xi16>, tensor<3x2xi8>, i16, i16) outs(%cst_1 : tensor<1x2xi16>) -> tensor<1x2xi16>
    return %4 : tensor<1x2xi16>
  }
}
