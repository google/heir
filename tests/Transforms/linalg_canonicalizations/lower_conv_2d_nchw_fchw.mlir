// RUN: heir-opt --linalg-canonicalizations %s | FileCheck %s

// CHECK: func @lower_conv_2d_nchw_fchw
// CHECK-SAME: (%[[IMAGE:.*]]: tensor<2x3x4x5xf32>, %[[FILTER:.*]]: tensor<6x3x2x2xf32>, %[[OUTPUT:.*]]: tensor<2x6x3x4xf32>)
func.func @lower_conv_2d_nchw_fchw(%image: tensor<2x3x4x5xf32>, %filter: tensor<6x3x2x2xf32>, %output: tensor<2x6x3x4xf32>) -> tensor<2x6x3x4xf32> {
  // CHECK-NEXT: %[[OUTER_LOOP:.*]] = affine.for %[[IV_N:.*]] = 0 to 2 iter_args(%[[ITER_ARG_N:.*]] = %[[OUTPUT]])
  // CHECK:        %[[MIDDLE_LOOP:.*]] = affine.for %[[IV_F:.*]] = 0 to 6 iter_args(%[[ITER_ARG_F:.*]] = %[[ITER_ARG_N]])
  // CHECK:          %[[OUTPUT_SLICE:.*]] = tensor.extract_slice %[[ITER_ARG_F]][%[[IV_N]], %[[IV_F]], 0, 0] [1, 1, 3, 4] [1, 1, 1, 1] : tensor<2x6x3x4xf32> to tensor<3x4xf32>
  // CHECK-NEXT:     %[[INNER_LOOP:.*]] = affine.for %[[IV_C:.*]] = 0 to 3 iter_args(%[[ITER_ARG_C:.*]] = %[[OUTPUT_SLICE]])
  // CHECK:            %[[IMAGE_SLICE:.*]] = tensor.extract_slice %[[IMAGE]][%[[IV_N]], %[[IV_C]], 0, 0] [1, 1, 4, 5] [1, 1, 1, 1] : tensor<2x3x4x5xf32> to tensor<4x5xf32>
  // CHECK-NEXT:       %[[FILTER_SLICE:.*]] = tensor.extract_slice %[[FILTER]][%[[IV_F]], %[[IV_C]], 0, 0] [1, 1, 2, 2] [1, 1, 1, 1] : tensor<6x3x2x2xf32> to tensor<2x2xf32>
  // CHECK-NEXT:       %[[EMPTY:.*]] = tensor.empty() : tensor<3x4xf32>
  // CHECK-NEXT:       %[[CONV:.*]] = linalg.conv_2d ins(%[[IMAGE_SLICE]], %[[FILTER_SLICE]] : tensor<4x5xf32>, tensor<2x2xf32>) outs(%[[EMPTY]] : tensor<3x4xf32>) -> tensor<3x4xf32>
  // CHECK-NEXT:       %[[ADD:.*]] = arith.addf %[[CONV]], %[[ITER_ARG_C]] : tensor<3x4xf32>
  // CHECK-NEXT:       affine.yield %[[ADD]]
  // CHECK:          %[[INSERT:.*]] = tensor.insert_slice %[[INNER_LOOP]] into %[[ITER_ARG_F]][%[[IV_N]], %[[IV_F]], 0, 0] [1, 1, 3, 4] [1, 1, 1, 1] : tensor<3x4xf32> into tensor<2x6x3x4xf32>
  // CHECK-NEXT:     affine.yield %[[INSERT]]
  // CHECK:        affine.yield %[[MIDDLE_LOOP]]
  // CHECK:      return %[[OUTER_LOOP]]
  %0 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
    ins(%image, %filter : tensor<2x3x4x5xf32>, tensor<6x3x2x2xf32>)
    outs(%output : tensor<2x6x3x4xf32>) -> tensor<2x6x3x4xf32>
  return %0 : tensor<2x6x3x4xf32>
}
