// RUN: heir-opt --lattigo-configure-crypto-context --split-input-file %s | FileCheck %s

// Test whether detection works for nested functions

!evaluator = !lattigo.bgv.evaluator
!params = !lattigo.bgv.parameter
!ct = !lattigo.rlwe.ciphertext

module attributes {scheme.bgv} {
  func.func @nested(%evaluator : !evaluator, %ct : !ct) -> !ct {
    %ct1 = lattigo.bgv.mul_new %evaluator, %ct, %ct : (!evaluator, !ct, !ct) -> !ct
    %res = lattigo.bgv.relinearize_new %evaluator, %ct1 : (!evaluator, !ct) -> !ct
    return %res : !ct
  }
  func.func @entry(%evaluator : !evaluator, %ct : !ct) -> !ct {
    %sub = call @nested(%evaluator, %ct) : (!evaluator, !ct) -> !ct
    return %sub : !ct
  }
}

// CHECK: @nested
// CHECK: @entry
// CHECK: @entry__configure
// CHECK: lattigo.rlwe.gen_relinearization_key
// CHECK-NOT: lattigo.rlwe.gen_galois_key
