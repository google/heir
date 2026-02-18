module {
  func.func public @just_relu(%arg: tensor<16xf32> {secret.secret}) -> tensor<16xf32> {
    %10 = call @relu(%arg) { domain_lower = -1.0, domain_upper = 1.0, degree = 25 } : (tensor<16xf32>) -> tensor<16xf32>
    return %10 : tensor<16xf32>
  }
  func.func private @relu(%arg0: tensor<16xf32>) -> tensor<16xf32> {
    %cst = arith.constant dense<0.000000e+00> : tensor<f32>
    %0 = tensor.empty() : tensor<16xf32>
    %broadcasted = linalg.broadcast ins(%cst : tensor<f32>) outs(%0 : tensor<16xf32>) dimensions = [0]
    %1 = tensor.empty() : tensor<16xf32>
    %mapped = linalg.map { arith.maximumf } ins(%arg0, %broadcasted : tensor<16xf32>, tensor<16xf32>) outs(%1 : tensor<16xf32>)
    return %mapped : tensor<16xf32>
  }
}
