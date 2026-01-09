// RUN: heir-opt --lattigo-alloc-to-inplace %s | FileCheck %s

!ct = !lattigo.rlwe.ciphertext
!decryptor = !lattigo.rlwe.decryptor
!encoder = !lattigo.bgv.encoder
!encryptor_pk = !lattigo.rlwe.encryptor<publicKey = true>
!evaluator = !lattigo.bgv.evaluator
!param = !lattigo.bgv.parameter
!pt = !lattigo.rlwe.plaintext
module attributes {bgv.schemeParam = #bgv.scheme_param<logN = 12, Q = [17179926529], P = [17179967489], plaintextModulus = 65537>, scheme.bgv} {
  // CHECK: func.func @add
  func.func @add(%evaluator: !evaluator, %param: !param, %encoder: !encoder, %ct: !ct) -> !ct {
    // CHECK-COUNT-3: lattigo.bgv.add
    // CHECK-NOT: lattigo.bgv.add_new
    %ct_0 = lattigo.bgv.add_new %evaluator, %ct, %ct : (!evaluator, !ct, !ct) -> !ct
    %ct_1 = lattigo.bgv.add_new %evaluator, %ct_0, %ct_0 : (!evaluator, !ct, !ct) -> !ct
    %ct_2 = lattigo.bgv.add_new %evaluator, %ct_1, %ct_1 : (!evaluator, !ct, !ct) -> !ct
    return %ct_2 : !ct
  }
}
