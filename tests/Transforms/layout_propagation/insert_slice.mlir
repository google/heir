// RUN: heir-opt --layout-propagation=ciphertext-size=32 --split-input-file %s | FileCheck %s

module {
  // CHECK: func.func @main
  // CHECK: tensor.insert_slice
  // CHECK: return
  func.func @main(%arg0: !secret.secret<tensor<4x4xf32>>) -> !secret.secret<tensor<1x2x4x4xf32>> {
    %0 = tensor.empty() : tensor<1x2x4x4xf32>
    %2 = secret.generic(%arg0: !secret.secret<tensor<4x4xf32>>) {
    ^body(%input0: tensor<4x4xf32>):
      %inserted_slice = tensor.insert_slice %input0 into %0[0, 0, 0, 0] [1, 1, 4, 4] [1, 1, 1, 1] : tensor<4x4xf32> into tensor<1x2x4x4xf32>
      %inserted_slice_1 = tensor.insert_slice %input0 into %inserted_slice[0, 1, 0, 0] [1, 1, 4, 4] [1, 1, 1, 1] : tensor<4x4xf32> into tensor<1x2x4x4xf32>
      %added = arith.addf %inserted_slice_1, %inserted_slice_1 : tensor<1x2x4x4xf32>
      secret.yield %added : tensor<1x2x4x4xf32>
    } -> !secret.secret<tensor<1x2x4x4xf32>>
    return %2 : !secret.secret<tensor<1x2x4x4xf32>>
  }
}

// -----

// By default the function arguments are assigned row major layouts, so
// convert_layout is inserted on the destination to convert the layout.
module {
  // CHECK: func.func @main
  // CHECK: tensor_ext.convert_layout
  // CHECK: tensor.insert_slice
  // CHECK: return
  func.func @main(%arg0: !secret.secret<tensor<4x4xf32>>, %arg1: !secret.secret<tensor<1x2x4x4xf32>>) -> !secret.secret<tensor<1x2x4x4xf32>> {
    %2 = secret.generic(%arg0: !secret.secret<tensor<4x4xf32>>, %arg1: !secret.secret<tensor<1x2x4x4xf32>>) {
    ^body(%input0: tensor<4x4xf32>, %input1: tensor<1x2x4x4xf32>):
      %inserted_slice = tensor.insert_slice %input0 into %input1[0, 0, 0, 0] [1, 1, 4, 4] [1, 1, 1, 1] : tensor<4x4xf32> into tensor<1x2x4x4xf32>
      secret.yield %inserted_slice : tensor<1x2x4x4xf32>
    } -> !secret.secret<tensor<1x2x4x4xf32>>
    return %2 : !secret.secret<tensor<1x2x4x4xf32>>
  }
}
