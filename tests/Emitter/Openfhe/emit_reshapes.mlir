// RUN: heir-translate %s --emit-openfhe-pke --split-input-file | FileCheck %s

!cc = !openfhe.crypto_context
!pk = !openfhe.public_key
!sk = !openfhe.private_key
module attributes {scheme.ckks} {
  // CHECK: std::vector<float> collapse_shape(
  // CHECK-SAME: std::vector<float> [[v0:.*]],
  // CHECK: return [[v0]];
  func.func @collapse_shape(%cc: !cc, %0: tensor<1x1024xf32>, %sk: !sk) -> tensor<1024xf32> {
    %collapsed = tensor.collapse_shape %0 [[0, 1]] : tensor<1x1024xf32> into tensor<1024xf32>
    return %collapsed : tensor<1024xf32>
  }
}

// -----

!cc = !openfhe.crypto_context
!pk = !openfhe.public_key
!sk = !openfhe.private_key
module attributes {scheme.ckks} {
  // CHECK: std::vector<float> expand_shape(
  // CHECK-SAME: std::vector<float> [[v0:.*]],
  // CHECK: return [[v0]];
  func.func @expand_shape(%cc: !cc, %0: tensor<1024xf32>, %sk: !sk) -> tensor<1x1024xf32> {
    %expanded = tensor.expand_shape %0 [[0, 1]] output_shape [1, 1024] : tensor<1024xf32> into tensor<1x1024xf32>
    return %expanded : tensor<1x1024xf32>
  }
}
