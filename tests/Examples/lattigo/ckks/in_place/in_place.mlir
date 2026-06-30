// force rerun
func.func @in_place(%arg0: tensor<8xf64> {secret.secret}) -> tensor<8xf64> {
  debug.validate %arg0 {name = "input"} : tensor<8xf64>
  %c1 = arith.constant 1 : index
  %0 = tensor_ext.rotate %arg0, %c1 : tensor<8xf64>, index
  debug.validate %0 {name = "rotate"} : tensor<8xf64>
  %cst = arith.constant dense<2.000000e+00> : tensor<8xf64>
  %1 = arith.mulf %0, %cst : tensor<8xf64>
  debug.validate %1 {name = "mul"} : tensor<8xf64>
  return %1 : tensor<8xf64>
}
