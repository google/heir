// RUN: heir-translate %s --emit-openfhe-pke | FileCheck %s

!cc = !openfhe.crypto_context
!pk = !openfhe.public_key
!sk = !openfhe.private_key
module attributes {scheme.ckks} {
  // CHECK: test_truncf_scalar({{.*}} [[arg0:[a-zA-Z0-9_]+]],
  func.func @test_truncf_scalar(%cc: !cc, %arg0: f64, %sk: !sk) -> f32 {
    // CHECK: float [[v1:.*]] = static_cast<float>([[arg0]]);
    // CHECK: return [[v1]];
    %0 = arith.truncf %arg0 : f64 to f32
    return %0 : f32
  }

  // CHECK: test_truncf_tensor({{.*}} [[arg1:[a-zA-Z0-9_]+]],
  func.func @test_truncf_tensor(%cc: !cc, %arg1: tensor<5xf64>, %sk: !sk) -> tensor<5xf32> {
    // CHECK: std::vector<float> [[v2:.*]](5);
    // CHECK: for (int i = 0; i < 5; ++i) {
    // CHECK:   [[v2]][i] = static_cast<float>([[arg1]][i]);
    // CHECK: }
    // CHECK: return [[v2]];
    %1 = arith.truncf %arg1 : tensor<5xf64> to tensor<5xf32>
    return %1 : tensor<5xf32>
  }
}
