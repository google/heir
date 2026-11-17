// RUN: not heir-translate %s --emit-openfhe-pke 2>&1 | FileCheck %s

module attributes {scheme.bgv} {
  func.func @test_2d_tensor_from_elements(%arg0: f32) -> tensor<1x1xf32> {
    // CHECK: 'tensor.from_elements' op expected a rank-0 or rank-1 tensor, but got rank 2
    %0 = tensor.from_elements %arg0 : tensor<1x1xf32>
    return %0 : tensor<1x1xf32>
  }
}
