// RUN: heir-opt --lattigo-configure-crypto-context --split-input-file %s | FileCheck %s

// Test whether detection works for one main function

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

// -----

// Test whether called function is skipped

!evaluator = !lattigo.bgv.evaluator
!params = !lattigo.bgv.parameter
!ct = !lattigo.rlwe.ciphertext

module attributes {scheme.bgv} {
  func.func @sub(%evaluator : !evaluator, %ct : !ct) -> !ct {
    %res = lattigo.bgv.add %evaluator, %ct, %ct : (!evaluator, !ct, !ct) -> !ct
    return %res : !ct
  }
  func.func @add(%evaluator : !evaluator, %ct : !ct) -> !ct {
    %res = lattigo.bgv.add %evaluator, %ct, %ct : (!evaluator, !ct, !ct) -> !ct
    %sub = call @sub(%evaluator, %res) : (!evaluator, !ct) -> !ct
    return %sub : !ct
  }
}

// CHECK: @sub
// CHECK: @add
// CHECK: @add__configure

// -----

// Test whether function declaration is skipped

!evaluator = !lattigo.bgv.evaluator
!params = !lattigo.bgv.parameter
!ct = !lattigo.rlwe.ciphertext

module attributes {scheme.bgv} {
  func.func private @sub(%evaluator : !evaluator, %ct : !ct) -> !ct
  func.func @add(%evaluator : !evaluator, %ct : !ct) -> !ct {
    %res = lattigo.bgv.add %evaluator, %ct, %ct : (!evaluator, !ct, !ct) -> !ct
    return %res : !ct
  }
}

// CHECK: @sub
// CHECK: @add
// CHECK: @add__configure
