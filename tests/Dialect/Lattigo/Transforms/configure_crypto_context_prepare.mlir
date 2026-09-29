// RUN: heir-opt --lattigo-configure-crypto-context=entry-function=prepare %s | FileCheck %s

// CHECK: func.func @prepare__configure
// CHECK-NOT: lattigo.rlwe.gen_relinearization_key
// CHECK: lattigo.rlwe.gen_galois_key
// CHECK-SAME: galoisElement = 5

!param = !lattigo.ckks.parameter
!encoder = !lattigo.ckks.encoder
!lt = !lattigo.ckks.linear_transformation

module attributes {scheme.ckks, ckks.schemeParam = #ckks.scheme_param<logN = 13, Q = [36028797018652673, 35184372121601], P = [1152921504606994433], logDefaultScale = 45>} {
  func.func @prepare(%param: !param, %encoder: !encoder) -> !lt {
    %diagonals = arith.constant dense<1.0> : tensor<2x4xf64>
    %res = lattigo.ckks.prepare_linear_transform %param, %encoder, %diagonals {diagonal_indices = array<i32: 0, 1>, levelQ = 0 : i64, logBabyStepGiantStepRatio = 0 : i64, logSlots = 2 : i64} : (!param, !encoder, tensor<2x4xf64>) -> !lt
    return %res : !lt
  }
}
