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
    %ct_0 = lattigo.ckks.linear_transform %evaluator, %encoder, %ct, %arg0 <diagonal_indices = [0, 1], levelQ = 5 : i32, logBabyStepGiantStepRatio = 2 : i64> : (!evaluator, !encoder, !ct, tensor<2x4096xf64>) -> !ct
    %ct_1 = lattigo.ckks.rotate_new %evaluator, %ct_0 <static_shift = 2048 : i32> : (!evaluator, !ct) -> !ct
    %ct_2 = lattigo.ckks.add_new %evaluator, %ct_1, %ct_0 : (!evaluator, !ct, !ct) -> !ct
    return %ct_2 : !ct
  }

  // CHECK: func Linear_transform_small
  // CHECK: ct1_slots := 1 << ct.LogDimensions.Cols
  // CHECK: if ct1_slots < 4 || ct1_slots % 4 != 0
  // CHECK: panic(fmt.Sprintf(
  // CHECK: if ct1_slots == 4
  // CHECK: ct1_diags[diagIndex] = v0[sourceRow*4:(sourceRow + 1)*4]
  // CHECK: } else {
  // CHECK: for j := 0; j < ct1_slots; j++
  // CHECK: diag[j] = v0[sourceRow*4 + (j % 4)]
  func.func @linear_transform_small(%evaluator: !evaluator, %param: !param, %encoder: !encoder, %ct: !ct, %arg0: tensor<2x4xf64>) -> !ct {
    %ct_0 = lattigo.ckks.linear_transform %evaluator, %encoder, %ct, %arg0 <diagonal_indices = [0, 1], levelQ = 5 : i32, logBabyStepGiantStepRatio = 2 : i64> : (!evaluator, !encoder, !ct, tensor<2x4xf64>) -> !ct
    return %ct_0 : !ct
  }
}

