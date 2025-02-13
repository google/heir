// RUN: heir-opt --lattigo-configure-crypto-context --verify-diagnostics %s

!evaluator = !lattigo.bgv.evaluator
!params = !lattigo.bgv.parameter
!ct = !lattigo.rlwe.ciphertext

// expected-warning@+1 {{Entry function not found, please provide entry-function in the pass options}}
module attributes {scheme.bgv} {
  func.func @__add(%evaluator : !evaluator, %ct : !ct) -> !ct {
    %res = lattigo.bgv.add_new %evaluator, %ct, %ct : (!evaluator, !ct, !ct) -> !ct
    return %res : !ct
  }
}
