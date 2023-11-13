// RUN: heir-opt --comb-to-cggi %s | FileCheck %s

module {
  // CHECK-NOT: secret
  // CHECK: @truth_table([[ARG:%.*]]: [[LWET:!lwe.lwe_ciphertext<.*>]]) -> [[LWET]]
  func.func @truth_table(%arg0: !secret.secret<i1>) -> !secret.secret<i1> {
    %0 = secret.generic
        ins(%arg0 : !secret.secret<i1>) {
        ^bb0(%ARG0: i1) :
            secret.yield %ARG0 : i1
        } -> (!secret.secret<i1>)
    // CHECK: return [[ARG]] : [[LWET]]
    func.return %0 : !secret.secret<i1>
  }
}
