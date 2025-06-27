func.func @matmul(%arg0: tensor<16xf32> {secret.secret}, %arg1: tensor<16xf32>, %arg2: tensor<16xf32>, %arg3: tensor<16xf32>, %arg4: tensor<16xf32>) -> tensor<16xf32> {
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %0 = arith.mulf %arg0, %arg1 : tensor<16xf32>
  %1 = tensor_ext.rotate %arg0, %c4 : tensor<16xf32>, index
  %2 = arith.mulf %1, %arg2 : tensor<16xf32>
  %3 = arith.addf %0, %2 : tensor<16xf32>
  %4 = arith.mulf %arg0, %arg3 : tensor<16xf32>
  %5 = arith.mulf %1, %arg4 : tensor<16xf32>
  %6 = arith.addf %4, %5 : tensor<16xf32>
  %7 = tensor_ext.rotate %6, %c8 : tensor<16xf32>, index
  %8 = arith.addf %3, %7 : tensor<16xf32>
  return %8 : tensor<16xf32>
}
