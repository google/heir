// RUN: heir-opt --lattigo-configure-crypto-context=entry-function=add %s | FileCheck %s

!evaluator = !lattigo.bgv.evaluator
!params = !lattigo.bgv.parameter
!ct = !lattigo.rlwe.ciphertext

module attributes {scheme.bgv} {
  func.func @add(%evaluator : !evaluator, %ct : !ct) -> !ct {
    %res = lattigo.bgv.add %evaluator, %ct, %ct : (!evaluator, !ct, !ct) -> !ct
    return %res : !ct
  }
}

// CHECK: @add
// CHECK: @add__configure
// CHECK-NOT: lattigo.rlwe.gen_relinearization_key
// CHECK-NOT: lattigo.rlwe.gen_galois_key
