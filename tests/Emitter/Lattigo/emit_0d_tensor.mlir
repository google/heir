// RUN: heir-translate %s --emit-lattigo | FileCheck %s

module attributes {scheme.bgv} {
  func.func @test_0d_tensor(%arg0: tensor<f32>) -> f32 {
    // CHECK: func test_0d_tensor([[arg0:[^ ]*]] []float32)
    // CHECK: [[v0:[^ ]*]] := [[arg0]][0]
    // CHECK: return [[v0]]
    %0 = tensor.extract %arg0[] : tensor<f32>
    return %0 : f32
  }

  func.func @test_0d_tensor_insert(%arg0: tensor<f32>, %v: f32) -> tensor<f32> {
    // CHECK: func test_0d_tensor_insert([[arg0:[^ ]*]] []float32, [[v:[^ ]*]] float32)
    // CHECK: [[v0:[^ ]*]] := append(make([]float32, 0, len([[arg0]])), [[arg0]]...)
    // CHECK: [[v0]][0] = [[v]]
    // CHECK: return [[v0]]
    %0 = tensor.insert %v into %arg0[] : tensor<f32>
    return %0 : tensor<f32>
  }
}
