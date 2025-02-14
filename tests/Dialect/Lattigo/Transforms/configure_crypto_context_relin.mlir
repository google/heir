// RUN: heir-opt --lattigo-configure-crypto-context=entry-function=relin %s | FileCheck %s

!evaluator = !lattigo.bgv.evaluator
!params = !lattigo.bgv.parameter
!ct = !lattigo.rlwe.ciphertext

module attributes {scheme.bgv} {
  func.func @relin(%evaluator : !evaluator, %ct : !ct) -> !ct {
    %ct1 = lattigo.bgv.mul %evaluator, %ct, %ct : (!evaluator, !ct, !ct) -> !ct
    %res = lattigo.bgv.relinearize %evaluator, %ct1 : (!evaluator, !ct) -> !ct
    return %res : !ct
  }
}

// CHECK: @relin
// CHECK: @relin__configure
// CHECK: lattigo.rlwe.gen_relinearization_key
// CHECK-NOT: lattigo.rlwe.gen_galois_key
