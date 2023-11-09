// RUN: heir-opt --comb-to-cggi %s | FileCheck %s

module {
  // CHECK: func.func @types([[ARG:%.*]]: [[T:!lwe.lwe_ciphertext<.*>]]) -> [[T]]
  func.func @types(%arg0: !secret.secret<i1>) -> !secret.secret<i1> {
    // CHECK: return [[ARG]] : [[T]]
    return %arg0 : !secret.secret<i1>
  }
}
