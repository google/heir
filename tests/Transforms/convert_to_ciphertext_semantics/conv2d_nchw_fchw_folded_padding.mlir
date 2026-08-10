// RUN: heir-opt --layout-propagation=min-slot-count=1024 %s | FileCheck %s --check-prefix=PROP
// RUN: heir-opt --layout-propagation=min-slot-count=1024 --convert-to-ciphertext-semantics=min-slot-count=1024 %s | FileCheck %s --check-prefix=CTS

// A "same" convolution: the 1x1 pad on (H, W) folds into the conv's own
// padding parameter, so the ciphertext stays packed at the unpadded 4x4 extent
// and the Toeplitz matrix is 64 x 64 instead of 64 x (4*6*6).

// PROP-NOT: tensor_ext.convert_layout
// PROP: linalg.conv_2d_nchw_fchw
// PROP-SAME: heir.conv_folded_padding = 1 : i64

// 64 diagonals, one per matrix row: the padded column count never leaked in.
// CTS: secret.generic
// CTS: tensor.extract_slice %{{.*}} : tensor<64x1024xf32> to tensor<1x1024xf32>

// A square matrix needs no squat-diagonal collapse, so the kernel ends at the
// last baby-step accumulation plus the bias. Sized against the unfolded
// 64 x 256 matrix it would rotate-and-add twice more first.
// CTS: %[[rot:.*]] = tensor_ext.rotate %{{.*}}, %c56
// CTS-NEXT: %[[sum:.*]] = arith.addf %{{.*}}, %[[rot]]
// CTS-NEXT: %[[biased:.*]] = arith.addf %[[sum]], %cst
// CTS-NEXT: secret.yield %[[biased]]

func.func @conv2d_padded(%arg0: !secret.secret<tensor<1x4x4x4xf32>>) -> !secret.secret<tensor<1x4x4x4xf32>> {
  %out = arith.constant dense<0.000000e+00> : tensor<1x4x4x4xf32>
  %filter = arith.constant dense<2.000000e+00> : tensor<4x4x3x3xf32>
  %0 = secret.generic(%arg0 : !secret.secret<tensor<1x4x4x4xf32>>) {
  ^body(%input0: tensor<1x4x4x4xf32>):
    %zero = arith.constant 0.000000e+00 : f32
    %padded = tensor.pad %input0 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%i: index, %j: index, %k: index, %l: index):
      tensor.yield %zero : f32
    } : tensor<1x4x4x4xf32> to tensor<1x4x6x6xf32>
    %1 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%padded, %filter : tensor<1x4x6x6xf32>, tensor<4x4x3x3xf32>) outs(%out : tensor<1x4x4x4xf32>) -> tensor<1x4x4x4xf32>
    secret.yield %1 : tensor<1x4x4x4xf32>
  } -> !secret.secret<tensor<1x4x4x4xf32>>
  return %0 : !secret.secret<tensor<1x4x4x4xf32>>
}
