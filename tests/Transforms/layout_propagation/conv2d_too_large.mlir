// RUN: heir-opt --layout-propagation=min-slot-count=1024 --verify-diagnostics %s

// The expanded filter matrix of this convolution has 16 * 16 * 16 = 4096 rows
// (one per output element), which does not fit in a single 1024-slot
// ciphertext. Previously the diagonal layout silently dropped the rows beyond
// 1024, and convert-to-ciphertext-semantics later failed with an unrealized
// conversion cast from 1 to 4 ciphertexts.
func.func @conv2d_too_large(%arg0: !secret.secret<tensor<1x3x16x16xf32>>) -> !secret.secret<tensor<1x16x16x16xf32>> {
  %cst = arith.constant dense<0.000000e+00> : tensor<1x16x16x16xf32>
  %filter = arith.constant dense<2.500000e-01> : tensor<16x3x1x1xf32>
  %0 = secret.generic(%arg0 : !secret.secret<tensor<1x3x16x16xf32>>) {
  ^body(%input0: tensor<1x3x16x16xf32>):
    // expected-error@+1 {{expanded filter matrix of shape 4096x768 does not fit in a single ciphertext of 1024 slots; multi-ciphertext convolutions are not yet supported. Set min-slot-count to at least 4096}}
    %1 = linalg.conv_2d_nchw_fchw
      { dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64> }
      ins(%input0, %filter : tensor<1x3x16x16xf32>, tensor<16x3x1x1xf32>)
      outs(%cst : tensor<1x16x16x16xf32>) -> tensor<1x16x16x16xf32>
    secret.yield %1 : tensor<1x16x16x16xf32>
  } -> !secret.secret<tensor<1x16x16x16xf32>>
  return %0 : !secret.secret<tensor<1x16x16x16xf32>>
}
