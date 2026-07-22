func.func @softmax(%arg0: tensor<8xf32> {secret.secret}) -> (tensor<8xf32> {secret.secret}) {
  %0 = math_ext.softmax %arg0 {domain_lower = -1.0 : f64, domain_upper = 1.0 : f64} : tensor<8xf32>
  return %0 : tensor<8xf32>
}
