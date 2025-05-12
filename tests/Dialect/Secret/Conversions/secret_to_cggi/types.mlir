// RUN: heir-opt --secret-to-cggi %s | FileCheck %s

module {
  // CHECK: func.func @types([[ARG:%.*]]: [[T:!lwe.lwe_ciphertext<.*>]]) -> [[T]]
  func.func @types(%arg0: !secret.secret<i1>) -> !secret.secret<i1> {
    // CHECK: return [[ARG]] : [[T]]
    return %arg0 : !secret.secret<i1>
  }

  // CHECK: func.func @multi([[ARG1:%.*]]: [[M:tensor<8x!lwe.*>>]]) -> [[M]]
  func.func @multi(%arg0: !secret.secret<i8>) -> !secret.secret<i8> {
    // CHECK: return [[ARG1]] : [[M]]
    return %arg0 :!secret.secret<i8>
  }
}
