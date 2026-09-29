// RUN: heir-opt --one-shot-bufferize="bufferize-function-boundaries" %s | FileCheck %s

!param = !lattigo.ckks.parameter
!encoder = !lattigo.ckks.encoder
!evaluator = !lattigo.ckks.evaluator
!ct = !lattigo.rlwe.ciphertext
!lt = !lattigo.ckks.linear_transformation

// CHECK: func.func @bufferize_prepare
// CHECK-SAME: ({{.*}}memref<2x4xf64{{.*}}>)
// CHECK: %[[RES:.*]] = lattigo.ckks.prepare_linear_transform %{{.*}}, %{{.*}}, %{{.*}} {diagonal_indices = array<i32: 0, 1>, levelQ = 0 : i64, logBabyStepGiantStepRatio = 0 : i64, logSlots = 2 : i64} : (!param, !encoder, memref<2x4xf64{{.*}}>) -> !linear_transformation
// CHECK: return %[[RES]]
func.func @bufferize_prepare(%param: !param, %encoder: !encoder, %diagonals: tensor<2x4xf64>) -> !lt {
  %res = lattigo.ckks.prepare_linear_transform %param, %encoder, %diagonals {diagonal_indices = array<i32: 0, 1>, levelQ = 0 : i64, logBabyStepGiantStepRatio = 0 : i64, logSlots = 2 : i64} : (!param, !encoder, tensor<2x4xf64>) -> !lt
  return %res : !lt
}

// CHECK: func.func @bufferize_linear_transform
// CHECK-SAME: ({{.*}}memref<2x4xf64{{.*}}>)
// CHECK: %[[RES:.*]] = lattigo.ckks.linear_transform %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} {diagonal_indices = array<i32: 0, 1>, levelQ = 0 : i64, logBabyStepGiantStepRatio = 0 : i64} : (!evaluator, !encoder, !ct, memref<2x4xf64{{.*}}>) -> !ct
// CHECK: return %[[RES]]
func.func @bufferize_linear_transform(%evaluator: !evaluator, %encoder: !encoder, %ct: !ct, %diagonals: tensor<2x4xf64>) -> !ct {
  %res = lattigo.ckks.linear_transform %evaluator, %encoder, %ct, %diagonals {diagonal_indices = array<i32: 0, 1>, levelQ = 0 : i64, logBabyStepGiantStepRatio = 0 : i64} : (!evaluator, !encoder, !ct, tensor<2x4xf64>) -> !ct
  return %res : !ct
}
