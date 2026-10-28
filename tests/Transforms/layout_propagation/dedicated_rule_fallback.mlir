// RUN: heir-opt --layout-propagation=min-slot-count=64 %s | FileCheck %s

// The tricyclic batch matmul layout needs coprime dimensions; for these
// shapes the op is laid out as a linalg.generic for the generic kernel.

// CHECK: func.func @batch_matmul
// CHECK-NOT: linalg.batch_matmul
// CHECK: linalg.generic
// CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel", "reduction"]
// CHECK: return
func.func @batch_matmul(%a: !secret.secret<tensor<2x3x4xf32>>, %b: !secret.secret<tensor<2x4x5xf32>>) -> !secret.secret<tensor<2x3x5xf32>> {
  %zero = arith.constant dense<0.0> : tensor<2x3x5xf32>
  %0 = secret.generic(%a: !secret.secret<tensor<2x3x4xf32>>, %b: !secret.secret<tensor<2x4x5xf32>>) {
  ^body(%x: tensor<2x3x4xf32>, %y: tensor<2x4x5xf32>):
    %1 = linalg.batch_matmul ins(%x, %y : tensor<2x3x4xf32>, tensor<2x4x5xf32>) outs(%zero : tensor<2x3x5xf32>) -> tensor<2x3x5xf32>
    secret.yield %1 : tensor<2x3x5xf32>
  } -> !secret.secret<tensor<2x3x5xf32>>
  return %0 : !secret.secret<tensor<2x3x5xf32>>
}

// The ciphertext-ciphertext tricyclic kernel rotates by up to
// h*n*p - 1 + h*m*(n-1) = 85 slots, which leaves too few of the 64 slots for
// the 40-element result, though every operand fits.

// CHECK: func.func @batch_matmul_reach
// CHECK-NOT: linalg.batch_matmul
// CHECK: linalg.generic
// CHECK: return
func.func @batch_matmul_reach(%a: !secret.secret<tensor<1x5x7xf32>>, %b: !secret.secret<tensor<1x7x8xf32>>) -> !secret.secret<tensor<1x5x8xf32>> {
  %zero = arith.constant dense<0.0> : tensor<1x5x8xf32>
  %0 = secret.generic(%a: !secret.secret<tensor<1x5x7xf32>>, %b: !secret.secret<tensor<1x7x8xf32>>) {
  ^body(%x: tensor<1x5x7xf32>, %y: tensor<1x7x8xf32>):
    %1 = linalg.batch_matmul ins(%x, %y : tensor<1x5x7xf32>, tensor<1x7x8xf32>) outs(%zero : tensor<1x5x8xf32>) -> tensor<1x5x8xf32>
    secret.yield %1 : tensor<1x5x8xf32>
  } -> !secret.secret<tensor<1x5x8xf32>>
  return %0 : !secret.secret<tensor<1x5x8xf32>>
}

// The data fits in one ciphertext, but the expanded 80x40 filter matrix does
// not.

// CHECK: func.func @conv1d_expanded_filter
// CHECK-NOT: linalg.conv_1d_ncw_fcw
// CHECK: linalg.generic
// CHECK: return
func.func @conv1d_expanded_filter(%x: !secret.secret<tensor<1x1x40xf32>>) -> !secret.secret<tensor<1x2x40xf32>> {
  %filter = arith.constant dense<0.5> : tensor<2x1x1xf32>
  %zero = arith.constant dense<0.0> : tensor<1x2x40xf32>
  %0 = secret.generic(%x: !secret.secret<tensor<1x1x40xf32>>) {
  ^body(%input: tensor<1x1x40xf32>):
    %1 = linalg.conv_1d_ncw_fcw {dilations = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} ins(%input, %filter : tensor<1x1x40xf32>, tensor<2x1x1xf32>) outs(%zero : tensor<1x2x40xf32>) -> tensor<1x2x40xf32>
    secret.yield %1 : tensor<1x2x40xf32>
  } -> !secret.secret<tensor<1x2x40xf32>>
  return %0 : !secret.secret<tensor<1x2x40xf32>>
}

// The conv kernel needs its data and result in one ciphertext each; these span
// two and four.

// CHECK: func.func @conv_multiple_ciphertexts
// CHECK-NOT: linalg.conv_2d_nchw_fchw
// CHECK: linalg.generic
// CHECK: return
func.func @conv_multiple_ciphertexts(%x: !secret.secret<tensor<1x2x8x8xf32>>) -> !secret.secret<tensor<1x4x6x6xf32>> {
  %filter = arith.constant dense<0.5> : tensor<4x2x3x3xf32>
  %zero = arith.constant dense<0.0> : tensor<1x4x6x6xf32>
  %0 = secret.generic(%x: !secret.secret<tensor<1x2x8x8xf32>>) {
  ^body(%input: tensor<1x2x8x8xf32>):
    %1 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%input, %filter : tensor<1x2x8x8xf32>, tensor<4x2x3x3xf32>) outs(%zero : tensor<1x4x6x6xf32>) -> tensor<1x4x6x6xf32>
    secret.yield %1 : tensor<1x4x6x6xf32>
  } -> !secret.secret<tensor<1x4x6x6xf32>>
  return %0 : !secret.secret<tensor<1x4x6x6xf32>>
}

// The conv kernel cannot read a strided conv's pixel-shuffled result.

// CHECK: func.func @conv_after_strided_conv
// CHECK: linalg.conv_2d_nchw_fchw
// CHECK-NOT: linalg.conv_2d_nchw_fchw
// CHECK: linalg.generic
// CHECK: return
func.func @conv_after_strided_conv(%x: !secret.secret<tensor<1x1x4x4xf32>>) -> !secret.secret<tensor<1x1x2x2xf32>> {
  %filter1 = arith.constant dense<0.5> : tensor<2x1x2x2xf32>
  %filter2 = arith.constant dense<0.5> : tensor<1x2x1x1xf32>
  %zero1 = arith.constant dense<0.0> : tensor<1x2x2x2xf32>
  %zero2 = arith.constant dense<0.0> : tensor<1x1x2x2xf32>
  %0 = secret.generic(%x: !secret.secret<tensor<1x1x4x4xf32>>) {
  ^body(%input: tensor<1x1x4x4xf32>):
    %1 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%input, %filter1 : tensor<1x1x4x4xf32>, tensor<2x1x2x2xf32>) outs(%zero1 : tensor<1x2x2x2xf32>) -> tensor<1x2x2x2xf32>
    %2 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%1, %filter2 : tensor<1x2x2x2xf32>, tensor<1x2x1x1xf32>) outs(%zero2 : tensor<1x1x2x2xf32>) -> tensor<1x1x2x2xf32>
    secret.yield %2 : tensor<1x1x2x2xf32>
  } -> !secret.secret<tensor<1x1x2x2xf32>>
  return %0 : !secret.secret<tensor<1x1x2x2xf32>>
}
