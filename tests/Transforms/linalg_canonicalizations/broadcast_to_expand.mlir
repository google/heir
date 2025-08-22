// RUN: heir-opt --linalg-canonicalizations --split-input-file %s | FileCheck %s

module {
  // CHECK: func @main
  // CHECK-SAME: %[[arg0:.*]]: tensor<512xf32>
  // CHECK: return %[[arg0]] : tensor<512xf32>
  func.func @main(%arg1 : tensor<512xf32>) -> (tensor<512xf32>) {
    %2 = tensor.empty() : tensor<1x512xf32>
    %broadcasted = linalg.broadcast ins(%arg1 : tensor<512xf32>) outs(%2 : tensor<1x512xf32>) dimensions = [0]
    %collapsed_1 = tensor.collapse_shape %broadcasted [[0, 1]] : tensor<1x512xf32> into tensor<512xf32>
    func.return %collapsed_1 : tensor<512xf32>
  }
}

// -----

module {
  // CHECK: func @no_match
  // CHECK-SAME: %[[arg0:.*]]: tensor<512xf32>
  // CHECK: %[[v0:.*]] = tensor.empty() : tensor<512x2xf32>
  // CHECK: %[[v1:.*]] = linalg.broadcast ins(%[[arg0]] : tensor<512xf32>) outs(%[[v0:.*]] : tensor<512x2xf32>) dimensions = [1]
  // CHECK: %[[v2:.*]] = tensor.collapse_shape %[[v1]] {{\[\[}}0, 1]] : tensor<512x2xf32> into tensor<1024xf32>
  // CHECK: return %[[v2]] : tensor<1024xf32>
  func.func @no_match(%arg1 : tensor<512xf32>) -> (tensor<1024xf32>) {
    %2 = tensor.empty() : tensor<512x2xf32>
    %broadcasted = linalg.broadcast ins(%arg1 : tensor<512xf32>) outs(%2 : tensor<512x2xf32>) dimensions = [1]
    %collapsed_1 = tensor.collapse_shape %broadcasted [[0, 1]] : tensor<512x2xf32> into tensor<1024xf32>
    func.return %collapsed_1 : tensor<1024xf32>
  }
}

// -----

module {
  // CHECK: func @main
  // CHECK-SAME: %[[arg0:.*]]: tensor<2x3xf32>
  // CHECK: return %[[arg0]] : tensor<2x3xf32>
  func.func @main(%arg1 : tensor<2x3xf32>) -> (tensor<2x3xf32>) {
    %2 = tensor.empty() : tensor<2x3x1xf32>
    %broadcasted = linalg.broadcast ins(%arg1 : tensor<2x3xf32>) outs(%2 : tensor<2x3x1xf32>) dimensions = [2]
    %collapsed_1 = tensor.collapse_shape %broadcasted [[0], [1, 2]] : tensor<2x3x1xf32> into tensor<2x3xf32>
    func.return %collapsed_1 : tensor<2x3xf32>
  }
}

// -----

module {
  // CHECK: func @main
  // CHECK-SAME: %[[arg0:.*]]: tensor<2x3xf32>
  // CHECK: %[[v0:.*]] = tensor.expand_shape %[[arg0]] {{\[\[}}0, 1], [2, 3]]
  // CHECK: return %[[v0]] : tensor<1x2x3x1xf32>
  func.func @main(%arg1 : tensor<2x3xf32>) -> (tensor<1x2x3x1xf32>) {
    %2 = tensor.empty() : tensor<1x2x3x1xf32>
    %broadcasted = linalg.broadcast ins(%arg1 : tensor<2x3xf32>) outs(%2 : tensor<1x2x3x1xf32>) dimensions = [0, 3]
    func.return %broadcasted : tensor<1x2x3x1xf32>
  }
}
