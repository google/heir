// RUN: heir-opt --lattigo-configure-crypto-context=entry-function=rotate %s | FileCheck %s

!evaluator = !lattigo.bgv.evaluator
!params = !lattigo.bgv.parameter
!ct = !lattigo.rlwe.ciphertext

module attributes {scheme.bgv} {
  func.func @rotate(%evaluator : !evaluator, %ct : !ct) -> !ct {
    %res = lattigo.bgv.rotate_columns_new %evaluator, %ct {static_shift = 1} : (!evaluator, !ct) -> !ct
    return %res : !ct
  }
}

// CHECK: @rotate
// CHECK: @rotate__configure
// CHECK-NOT: lattigo.rlwe.gen_relinearization_key
// CHECK: lattigo.rlwe.gen_galois_key
