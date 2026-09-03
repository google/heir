// RUN: heir-opt --layout-propagation=min-slot-count=1024 %s | FileCheck %s

// A stride-2 conv leaves its result in a gapped packing. The second conv pads
// that result, so its data operand is neither row major nor unpadded. Both convs
// must still fold their pad into their own `padding` parameter, and neither may
// pay for an online layout conversion: the second conv reads the gapped packing
// where it sits by absorbing it into its plaintext diagonal filter.

// The second conv absorbs, so its filter layout is built at the ciphertext
// width rather than at the Toeplitz C*W. It must record that width, because the
// kernel folds its partial sums over it.

// CHECK-NOT: tensor_ext.convert_layout
// CHECK: linalg.conv_1d_ncw_fcw
// CHECK-SAME: heir.conv_folded_padding = 1 : i64
// CHECK-NOT: tensor_ext.convert_layout
// CHECK: linalg.conv_1d_ncw_fcw
// CHECK-SAME: heir.absorbed_matrix_width = 1024 : i64
// CHECK-SAME: heir.conv_folded_padding = 1 : i64
// CHECK-NOT: tensor_ext.convert_layout

func.func @conv1d_gapped_folded_padding(
    %arg0: !secret.secret<tensor<1x8x8xf32>>) -> !secret.secret<tensor<1x8x4xf32>> {
  %out = arith.constant dense<0.000000e+00> : tensor<1x8x4xf32>
  %filter1 = arith.constant dense<2.000000e+00> : tensor<8x8x3xf32>
  %filter2 = arith.constant dense<3.000000e+00> : tensor<8x8x3xf32>
  %0 = secret.generic(%arg0 : !secret.secret<tensor<1x8x8xf32>>) {
  ^body(%input0: tensor<1x8x8xf32>):
    %zero = arith.constant 0.000000e+00 : f32
    %padded0 = tensor.pad %input0 low[0, 0, 1] high[0, 0, 1] {
    ^bb0(%i: index, %j: index, %k: index):
      tensor.yield %zero : f32
    } : tensor<1x8x8xf32> to tensor<1x8x10xf32>
    %1 = linalg.conv_1d_ncw_fcw {dilations = dense<1> : vector<1xi64>, strides = dense<2> : vector<1xi64>} ins(%padded0, %filter1 : tensor<1x8x10xf32>, tensor<8x8x3xf32>) outs(%out : tensor<1x8x4xf32>) -> tensor<1x8x4xf32>
    %padded1 = tensor.pad %1 low[0, 0, 1] high[0, 0, 1] {
    ^bb0(%i: index, %j: index, %k: index):
      tensor.yield %zero : f32
    } : tensor<1x8x4xf32> to tensor<1x8x6xf32>
    %2 = linalg.conv_1d_ncw_fcw {dilations = dense<1> : vector<1xi64>, strides = dense<1> : vector<1xi64>} ins(%padded1, %filter2 : tensor<1x8x6xf32>, tensor<8x8x3xf32>) outs(%out : tensor<1x8x4xf32>) -> tensor<1x8x4xf32>
    secret.yield %2 : tensor<1x8x4xf32>
  } -> !secret.secret<tensor<1x8x4xf32>>
  return %0 : !secret.secret<tensor<1x8x4xf32>>
}
