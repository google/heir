// From https://github.com/google/heir/pull/1182

func.func @add(%arg0: tensor<4xi16> {secret.secret}, %arg1: tensor<4xi16> {secret.secret}) -> tensor<4xi16> {
  %0 = arith.addi %arg0, %arg1 : tensor<4xi16>
  %1 = arith.muli %0, %arg1 : tensor<4xi16>
  %c1 = arith.constant 1 : index
  %2 = tensor_ext.rotate %1, %c1 : tensor<4xi16>, index
  return %2 : tensor<4xi16>
}
