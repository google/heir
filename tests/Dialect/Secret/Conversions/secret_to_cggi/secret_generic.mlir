// RUN: heir-opt --secret-distribute-generic --secret-to-cggi %s | FileCheck %s

// CHECK: ![[ct_ty:.*]] = !lwe.new_lwe_ciphertext

module {
  // CHECK-NOT: secret
  // CHECK: @truth_table([[ARG:%.*]]: ![[ct_ty]]) -> ![[ct_ty]]
  func.func @truth_table(%arg0: !secret.secret<i1>) -> !secret.secret<i1> {
    %0 = secret.generic
        (%arg0 : !secret.secret<i1>) {
        ^bb0(%ARG0: i1) :
            secret.yield %ARG0 : i1
        } -> (!secret.secret<i1>)
    // CHECK: return [[ARG]] : ![[ct_ty]]
    func.return %0 : !secret.secret<i1>
  }
}
