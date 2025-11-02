// RUN: heir-opt --forward-insert-slice-to-extract-slice --cse %s | FileCheck %s

module {
  // CHECK: func.func @convolution
  func.func @convolution(%arg0: tensor<1x1x4x4xf32>, %arg1: tensor<2x1x3x3xf32>) -> tensor<1x2x2x2xf32> {
    // CHECK: %[[empty:.*]] = tensor.empty() : tensor<2x2xf32>
    %0 = tensor.empty() : tensor<1x2x2x2xf32>
    %extracted_slice = tensor.extract_slice %arg1[0, 0, 0, 0] [1, 1, 3, 3] [1, 1, 1, 1] : tensor<2x1x3x3xf32> to tensor<3x3xf32>
    %1 = tensor.empty() : tensor<2x2xf32>
    %extracted_slice_0 = tensor.extract_slice %arg1[1, 0, 0, 0] [1, 1, 3, 3] [1, 1, 1, 1] : tensor<2x1x3x3xf32> to tensor<3x3xf32>
    %extracted_slice_1 = tensor.extract_slice %0[0, 0, 0, 0] [1, 1, 2, 2] [1, 1, 1, 1] : tensor<1x2x2x2xf32> to tensor<2x2xf32>
    %extracted_slice_2 = tensor.extract_slice %arg0[0, 0, 0, 0] [1, 1, 4, 4] [1, 1, 1, 1] : tensor<1x1x4x4xf32> to tensor<4x4xf32>
    %2 = linalg.conv_2d ins(%extracted_slice_2, %extracted_slice : tensor<4x4xf32>, tensor<3x3xf32>) outs(%1 : tensor<2x2xf32>) -> tensor<2x2xf32>
    // CHECK: linalg.conv_2d
    // %extracted_slice_1 is empty, so ensure this uses an empty tensor.
    // CHECK: arith.addf {{.*}}, %[[empty]] : tensor<2x2xf32>
    %3 = arith.addf %2, %extracted_slice_1 : tensor<2x2xf32>
    %inserted_slice = tensor.insert_slice %3 into %0[0, 0, 0, 0] [1, 1, 2, 2] [1, 1, 1, 1] : tensor<2x2xf32> into tensor<1x2x2x2xf32>
    %extracted_slice_3 = tensor.extract_slice %inserted_slice[0, 1, 0, 0] [1, 1, 2, 2] [1, 1, 1, 1] : tensor<1x2x2x2xf32> to tensor<2x2xf32>
    %4 = linalg.conv_2d ins(%extracted_slice_2, %extracted_slice_0 : tensor<4x4xf32>, tensor<3x3xf32>) outs(%1 : tensor<2x2xf32>) -> tensor<2x2xf32>
    // CHECK: linalg.conv_2d
    // %extracted_slice_3 is empty as well.
    // CHECK: arith.addf {{.*}}, %[[empty]] : tensor<2x2xf32>
    %5 = arith.addf %4, %extracted_slice_3 : tensor<2x2xf32>
    %inserted_slice_4 = tensor.insert_slice %5 into %inserted_slice[0, 1, 0, 0] [1, 1, 2, 2] [1, 1, 1, 1] : tensor<2x2xf32> into tensor<1x2x2x2xf32>
    return %inserted_slice_4 : tensor<1x2x2x2xf32>
  }
}
