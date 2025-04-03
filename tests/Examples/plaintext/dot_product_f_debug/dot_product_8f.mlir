func.func @dot_product(%arg0: tensor<8xf32> {secret.secret}, %arg1: tensor<8xf32> {secret.secret}) -> f32 {
  %c0 = arith.constant 0 : index
  %c0_sf32 = arith.constant 0.1 : f32
  %0 = affine.for %arg2 = 0 to 8 iter_args(%iter = %c0_sf32) -> (f32) {
    %1 = tensor.extract %arg0[%arg2] : tensor<8xf32>
    %2 = tensor.extract %arg1[%arg2] : tensor<8xf32>
    %3 = arith.mulf %1, %2 : f32
    %4 = arith.addf %iter, %3 : f32
    affine.yield %4 : f32
  }
  return %0 : f32
}
