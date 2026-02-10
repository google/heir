// RUN: heir-translate %s --emit-openfhe-pke | FileCheck %s

!cc = !openfhe.crypto_context
!pk = !openfhe.public_key
!sk = !openfhe.private_key
module attributes {scheme.ckks} {
  func.func @tensor_addf(%cc: !cc, %0: tensor<5xf32>, %sk: !sk) -> tensor<5xf32> {
    // CHECK: std::vector<float> [[v1:.*]](5);
    // CHECK: for (int i = 0; i < 5; ++i) {
    // CHECK:   [[v1]][i] =
    // CHECK: }
    // CHECK: return [[v1]];
    %1 = arith.addf %0, %0 : tensor<5xf32>
    return %1 : tensor<5xf32>
  }
  func.func @tensor_mulf(%cc: !cc, %0: tensor<5xf32>, %sk: !sk) -> tensor<5xf32> {
    // CHECK: std::vector<float> [[v1:.*]](5);
    // CHECK: for (int i = 0; i < 5; ++i) {
    // CHECK:   [[v1]][i] =
    // CHECK: }
    // CHECK: return [[v1]];
    %1 = arith.mulf %0, %0 : tensor<5xf32>
    return %1 : tensor<5xf32>
  }
}
