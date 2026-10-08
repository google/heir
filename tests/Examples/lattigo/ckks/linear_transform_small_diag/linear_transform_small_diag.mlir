!ct = !lattigo.rlwe.ciphertext
!encoder = !lattigo.ckks.encoder
!evaluator = !lattigo.ckks.evaluator
!param = !lattigo.ckks.parameter
!lt = !lattigo.ckks.linear_transformation

module attributes {scheme.ckks, ckks.schemeParam = #ckks.scheme_param<logN = 13, Q = [536903681, 67043329, 66994177, 67239937], P = [536952833, 536690689], logDefaultScale = 26>} {
  func.func @test_small_diag(%evaluator: !evaluator, %param: !param, %encoder: !encoder, %ct: !ct, %diagonals: tensor<4x4xf64>) -> !ct {
    %lt = lattigo.ckks.prepare_linear_transform %param, %encoder, %diagonals <
      diagonal_indices = [-1, 0, 1, 2],
      levelQ = 3 : i64,
      logSlots = 12 : i64,
      logBabyStepGiantStepRatio = 1 : i64
    > : (!param, !encoder, tensor<4x4xf64>) -> !lt
    %out = lattigo.ckks.apply_linear_transform %evaluator, %ct, %lt : (!evaluator, !ct, !lt) -> !ct
    return %out : !ct
  }

  func.func @test_prepare(%param: !param, %encoder: !encoder, %diagonals: tensor<4x4xf64>) -> !lt {
    %lt = lattigo.ckks.prepare_linear_transform %param, %encoder, %diagonals <
      diagonal_indices = [-1, 0, 1, 2],
      levelQ = 3 : i64,
      logSlots = 12 : i64,
      logBabyStepGiantStepRatio = 1 : i64
    > : (!param, !encoder, tensor<4x4xf64>) -> !lt
    return %lt : !lt
  }

  func.func @test_apply(%evaluator: !evaluator, %ct: !ct, %lt: !lt) -> !ct {
    %out = lattigo.ckks.apply_linear_transform %evaluator, %ct, %lt : (!evaluator, !ct, !lt) -> !ct
    return %out : !ct
  }

  func.func @test_full_diag(%param: !param, %encoder: !encoder, %diagonals: tensor<2x4096xf64>) -> !lt {
    %lt = lattigo.ckks.prepare_linear_transform %param, %encoder, %diagonals <
      diagonal_indices = [0, 1],
      levelQ = 3 : i64,
      logSlots = 12 : i64,
      logBabyStepGiantStepRatio = 1 : i64
    > : (!param, !encoder, tensor<2x4096xf64>) -> !lt
    return %lt : !lt
  }
}
