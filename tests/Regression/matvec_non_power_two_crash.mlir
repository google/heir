// RUN: heir-opt %s --annotate-module="backend=lattigo scheme=ckks" --mlir-to-ckks="ciphertext-degree=1024 modulus-switch-after-mul=true experimental-disable-loop-unroll=true level-budget=40 first-mod-bits=55" --scheme-to-lattigo
// TODO(#2960): remove XFAIL and update test appropriately
// XFAIL: *

module {
  func.func @matvec(%arg0 : tensor<33xf32> {secret.secret}, %matrix: tensor<32x33xf32>) -> tensor<32xf32> {
    %out = arith.constant dense<0.0> : tensor<32xf32>
    %0 = linalg.matvec ins(%matrix, %arg0 : tensor<32x33xf32>, tensor<33xf32>) outs(%out : tensor<32xf32>) -> tensor<32xf32>
    return %0 : tensor<32xf32>
  }
}
