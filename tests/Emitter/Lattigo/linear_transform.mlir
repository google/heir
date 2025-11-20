// RUN: heir-translate %s --emit-lattigo | FileCheck %s

// CHECK: lintrans.Diagonals
// CHECK: lintrans.Parameters
// CHECK: lintrans.NewTransformation
// CHECK: lintrans.Encode
// CHECK: lintrans.NewEvaluator
// CHECK: EvaluateNew

!ct = !lattigo.rlwe.ciphertext
!encoder = !lattigo.ckks.encoder
!evaluator = !lattigo.ckks.evaluator
!param = !lattigo.ckks.parameter
module attributes {scheme.ckks} {
  func.func @linear_transform(%evaluator: !evaluator, %param: !param, %encoder: !encoder, %ct: !ct, %arg0: tensor<2x4096xf64>) -> !ct {
    %ct_0 = lattigo.ckks.linear_transform %evaluator, %encoder, %ct, %arg0 {levelQ = 5 : i32, logBabyStepGiantStepRatio = 2 : i64, diagonal_indices = array<i32: 0, 1>} : (!evaluator, !encoder, !ct, tensor<2x4096xf64>) -> !ct
    %ct_1 = lattigo.ckks.rotate_new %evaluator, %ct_0 {offset = 2048 : i32} : (!evaluator, !ct) -> !ct
    %ct_2 = lattigo.ckks.add_new %evaluator, %ct_1, %ct_0 : (!evaluator, !ct, !ct) -> !ct
    return %ct_2 : !ct
  }
}
