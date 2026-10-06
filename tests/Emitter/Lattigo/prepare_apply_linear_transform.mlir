// RUN: heir-translate %s --emit-lattigo | FileCheck %s

!ct = !lattigo.rlwe.ciphertext
!encoder = !lattigo.ckks.encoder
!evaluator = !lattigo.ckks.evaluator
!param = !lattigo.ckks.parameter
!lt = !lattigo.ckks.linear_transformation

module attributes {scheme.ckks} {
  // CHECK: func Test_prepare
  // CHECK-NOT: linear_transformation_slots :=
  // CHECK: linear_transformation_diags[diagIndex] = v0[sourceRow*4096:(sourceRow + 1)*4096]
  // CHECK: lintrans.Parameters
  // CHECK: lintrans.NewTransformation
  // CHECK: lintrans.Encode
  func.func @test_prepare(%param: !param, %encoder: !encoder, %arg0: tensor<2x4096xf64>) -> !lt {
    %lt = lattigo.ckks.prepare_linear_transform %param, %encoder, %arg0 <
      diagonal_indices = [0, 1],
      levelQ = 5,
      logSlots = 12,
      logBabyStepGiantStepRatio = 2
    > : (!param, !encoder, tensor<2x4096xf64>) -> !lt
    return %lt : !lt
  }

  // CHECK: func Test_apply
  // CHECK: lintrans.NewEvaluator
  // CHECK: EvaluateNew
  func.func @test_apply(%evaluator: !evaluator, %ct: !ct, %lt: !lt) -> !ct {
    %ct_out = lattigo.ckks.apply_linear_transform %evaluator, %ct, %lt : (!evaluator, !ct, !lt) -> !ct
    return %ct_out : !ct
  }

  // CHECK: func Test_memref_lt
  func.func @test_memref_lt(%storage: memref<2x!lt>) -> !lt {
    %c0 = arith.constant 0 : index
    %lt = memref.load %storage[%c0] : memref<2x!lt>
    return %lt : !lt
  }

  // CHECK: func Test_prepare_small
  // CHECK: linear_transformation_slots := 1 << 12
  // CHECK: diag := make([]float64, linear_transformation_slots)
  // CHECK: for j := 0; j < linear_transformation_slots; j++
  // CHECK: diag[j] = v0[sourceRow*4 + (j % 4)]
  // CHECK: linear_transformation_diags[diagIndex] = diag
  func.func @test_prepare_small(%param: !param, %encoder: !encoder, %arg0: tensor<2x4xf64>) -> !lt {
    %lt = lattigo.ckks.prepare_linear_transform %param, %encoder, %arg0 <
      diagonal_indices = [0, 1],
      levelQ = 5,
      logSlots = 12,
      logBabyStepGiantStepRatio = 2
    > : (!param, !encoder, tensor<2x4xf64>) -> !lt
    return %lt : !lt
  }
}
