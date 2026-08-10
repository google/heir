// RUN: heir-opt --layout-propagation=min-slot-count=1024 %s | FileCheck %s

// A `heir.conv_folded_padding` already on the input — left by an earlier run of
// this pass, or preserved across an op clone — must be cleared when this run
// folds nothing. Left in place it would make ConvertToCiphertextSemantics size
// the Toeplitz matrix against an operand 2*p smaller on each spatial dim than
// the one the ciphertext actually holds.

// CHECK: linalg.conv_2d_nchw_fchw
// CHECK-NOT: heir.conv_folded_padding

func.func @stale_folded_padding(%arg0: !secret.secret<tensor<1x4x4x4xf32>>) -> !secret.secret<tensor<1x4x4x4xf32>> {
  %out = arith.constant dense<0.000000e+00> : tensor<1x4x4x4xf32>
  %filter = arith.constant dense<3.000000e+00> : tensor<4x4x1x1xf32>
  %0 = secret.generic(%arg0 : !secret.secret<tensor<1x4x4x4xf32>>) {
  ^body(%input0: tensor<1x4x4x4xf32>):
    // No tensor.pad to fold, but the op arrives already carrying the attribute.
    %1 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>, heir.conv_folded_padding = 1 : i64} ins(%input0, %filter : tensor<1x4x4x4xf32>, tensor<4x4x1x1xf32>) outs(%out : tensor<1x4x4x4xf32>) -> tensor<1x4x4x4xf32>
    secret.yield %1 : tensor<1x4x4x4xf32>
  } -> !secret.secret<tensor<1x4x4x4xf32>>
  return %0 : !secret.secret<tensor<1x4x4x4xf32>>
}
