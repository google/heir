// RUN: heir-opt --lattigo-alloc-to-inplace %s | FileCheck %s

// alloc-to-place should work with input with mixed AllocOp and InplaceOp

!ct = !lattigo.rlwe.ciphertext
!encoder = !lattigo.bgv.encoder
!evaluator = !lattigo.bgv.evaluator
!param = !lattigo.bgv.parameter
!pt = !lattigo.rlwe.plaintext

// CHECK-LABEL: func.func @add
func.func @add(%evaluator: !evaluator, %param: !param, %encoder: !encoder, %ct: !ct) -> !ct {
  // no new allocation found
  // CHECK-NOT: _new
  %ct_0 = lattigo.bgv.add %evaluator, %ct, %ct, %ct : (!evaluator, !ct, !ct, !ct) -> !ct
  %ct_1 = lattigo.bgv.add_new %evaluator, %ct_0, %ct_0 : (!evaluator, !ct, !ct) -> !ct
  %ct_2 = lattigo.bgv.add %evaluator, %ct_1, %ct_1, %ct_1 : (!evaluator, !ct, !ct, !ct) -> !ct
  %ct_3 = lattigo.bgv.add_new %evaluator, %ct_2, %ct_2 : (!evaluator, !ct, !ct) -> !ct
  %ct_4 = lattigo.bgv.add %evaluator, %ct_3, %ct_3, %ct_3 : (!evaluator, !ct, !ct, !ct) -> !ct
  %ct_5 = lattigo.bgv.add_new %evaluator, %ct_4, %ct_4 : (!evaluator, !ct, !ct) -> !ct
  return %ct_5 : !ct
}
