// RUN: heir-translate %s --emit-lattigo | FileCheck %s

// The preparation builds and encodes the transformation from compile-time
// parameters (no ciphertext reads); the application only evaluates it.

// CHECK: func Prepare(param ckks.Parameters, encoder *ckks.Encoder) (lintrans.LinearTransformation)
// CHECK: [[DIAGONALS:v[0-9]+]] := slices.Repeat
// CHECK: [[ROWS:[a-z_]+_source_rows]] := []int{1, 3}
// CHECK: lintrans.Diagonals
// CHECK: [[DIAGONALS]][[[ROWS]][i]*4096:([[ROWS]][i]+1)*4096]
// CHECK: lintrans.Parameters
// CHECK: LevelQ: 5,
// CHECK: Scale: rlwe.NewScale(param.GetRLWEParameters().Q()[5]),
// CHECK: LogDimensions: ring.Dimensions{Rows: 0, Cols: 12},
// CHECK: lintrans.NewTransformation(param.GetRLWEParameters()
// CHECK: lintrans.Encode

// CHECK: func Apply(evaluator *ckks.Evaluator, ct *rlwe.Ciphertext, linear_transformation lintrans.LinearTransformation) (*rlwe.Ciphertext)
// CHECK-NOT: lintrans.NewTransformation
// CHECK: lintrans.NewEvaluator(evaluator)
// CHECK: EvaluateNew(ct, linear_transformation)

!ct = !lattigo.rlwe.ciphertext
!encoder = !lattigo.ckks.encoder
!evaluator = !lattigo.ckks.evaluator
!param = !lattigo.ckks.parameter
!lt = !lattigo.ckks.linear_transformation
module attributes {scheme.ckks} {
  func.func @prepare(%param: !param, %encoder: !encoder) -> !lt {
    %diagonals = arith.constant dense<1.0> : tensor<4x4096xf64>
    %lt0 = lattigo.ckks.prepare_linear_transform %param, %encoder, %diagonals {diagonal_indices = array<i32: 0, 2>, source_row_indices = array<i32: 1, 3>, levelQ = 5 : i64, logSlots = 12 : i64, logBabyStepGiantStepRatio = 0 : i64} : (!param, !encoder, tensor<4x4096xf64>) -> !lt
    return %lt0 : !lt
  }
  func.func @apply(%evaluator: !evaluator, %ct0: !ct, %lt0: !lt) -> !ct {
    %ct1 = lattigo.ckks.apply_linear_transform %evaluator, %ct0, %lt0 : (!evaluator, !ct, !lt) -> !ct
    return %ct1 : !ct
  }
}
