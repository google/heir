// RUN: heir-opt --layout-propagation %s | FileCheck %s

// The correctness of the collapse_shape layout is unit tested in
// lib/Utils/Layout/UtilsTest.cpp

module {
  // CHECK: func.func @main
  // CHECK: tensor.collapse_shape
  // CHECK: return
  func.func @main(%arg0: !secret.secret<tensor<16x5x5xf32>>) -> !secret.secret<tensor<400xf32>> {
    %2 = secret.generic(%arg0: !secret.secret<tensor<16x5x5xf32>>) {
    ^body(%input0: tensor<16x5x5xf32>):
      %collapsed_52 = tensor.collapse_shape %input0 [[0, 1, 2]] : tensor<16x5x5xf32> into tensor<400xf32>
      secret.yield %collapsed_52 : tensor<400xf32>
    } -> !secret.secret<tensor<400xf32>>
    return %2 : !secret.secret<tensor<400xf32>>
  }
}
