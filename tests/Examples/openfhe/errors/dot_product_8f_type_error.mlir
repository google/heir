// RUN: heir-opt --verify-diagnostics --mlir-to-bgv='ciphertext-degree=8' --scheme-to-openfhe='entry-function=dot_product' %s

// expected-error@below {{Floating point types are not supported in BGV. Maybe you meant to use a CKKS pipeline like --mlir-to-ckks?}}
func.func @dot_product(%arg0: tensor<8xf16> {secret.secret}, %arg1: tensor<8xf16> {secret.secret}) -> f16 {
  %c0 = arith.constant 0 : index
  %c0_sf16 = arith.constant 0.1 : f16
  %0 = affine.for %arg2 = 0 to 8 iter_args(%iter = %c0_sf16) -> (f16) {
    %1 = tensor.extract %arg0[%arg2] : tensor<8xf16>
    %2 = tensor.extract %arg1[%arg2] : tensor<8xf16>
    %3 = arith.mulf %1, %2 : f16
    %4 = arith.addf %iter, %3 : f16
    affine.yield %4 : f16
  }
  return %0 : f16
}
