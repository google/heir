// RUN: heir-translate %s --emit-openfhe-pke | FileCheck %s

!cc = !openfhe.crypto_context
!pt = !openfhe.plaintext
!pk = !openfhe.public_key
!ct = !openfhe.ciphertext

module attributes {scheme.bgv} {
  func.func @test_0d_tensor(%arg0: tensor<f32>) -> f32 {
    // CHECK: float test_0d_tensor(std::vector<float> [[arg0:[^ ]*]])
    // CHECK: float [[v0:[^ ]*]] = [[arg0]][0];
    // CHECK: return [[v0]];
    %0 = tensor.extract %arg0[] : tensor<f32>
    return %0 : f32
  }

  func.func @test_0d_tensor_insert(%arg0: tensor<f32>, %v: f32) -> tensor<f32> {
    // CHECK: std::vector<float> test_0d_tensor_insert(std::vector<float> [[arg0:[^ ]*]], float [[v:[^ ]*]])
    // CHECK: [[arg0]][0] = [[v]];
    // CHECK: return [[arg0]];
    %0 = tensor.insert %v into %arg0[] : tensor<f32>
    return %0 : tensor<f32>
  }
}
