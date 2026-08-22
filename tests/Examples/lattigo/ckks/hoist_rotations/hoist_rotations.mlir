func.func @hoist_rotations(%arg0: tensor<8xf64> {secret.secret}) -> tensor<8xf64> {
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %r1 = tensor_ext.rotate %arg0, %c1 : tensor<8xf64>, index
  %r3 = tensor_ext.rotate %arg0, %c3 : tensor<8xf64>, index
  %add = arith.addf %r1, %r3 : tensor<8xf64>
  return %add : tensor<8xf64>
}
