// RUN: heir-opt --lwe-to-lattigo %s | FileCheck %s

#inverse_canonical_encoding = #lwe.inverse_canonical_encoding<scaling_factor = 45>
#key = #lwe.key<>
#modulus_chain = #lwe.modulus_chain<elements = <36028797018652673 : i64, 35184372121601 : i64>, current = 0>
#ring_f64_1_x1024 = #polynomial.ring<coefficientType = f64, polynomialModulus = <1 + x**1024>>
!rns_L0 = !rns.rns<!mod_arith.int<36028797018652673 : i64>>
#ring_rns_L0_1_x1024 = #polynomial.ring<coefficientType = !rns_L0, polynomialModulus = <1 + x**1024>>
#ciphertext_space_L0 = #lwe.ciphertext_space<ring = #ring_rns_L0_1_x1024, encryption_type = mix>
!ct = !lwe.lwe_ciphertext<plaintext_space = <ring = #ring_f64_1_x1024, encoding = #inverse_canonical_encoding>, ciphertext_space = #ciphertext_space_L0, key = #key, modulus_chain = #modulus_chain>
!prepared = !kernel.prepared_linear_transform<level = 0, slots = 512, log_bsgs_ratio = 0>

// CHECK-DAG: !ct = !lattigo.rlwe.ciphertext
// CHECK-DAG: !linear_transformation = !lattigo.ckks.linear_transformation

module attributes {backend.lattigo, ckks.schemeParam = #ckks.scheme_param<logN = 13, Q = [36028797018652673, 35184372121601], P = [1152921504606994433], logDefaultScale = 45, encryptionTechnique = extended>, scheme.ckks} {
  // CHECK: func.func @test_prepare_apply
  // CHECK: %[[DIAGONALS:.*]] = arith.constant dense<{{\[\[}}1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00], [5.000000e+00, 6.000000e+00], [7.000000e+00, 8.000000e+00{{\]\]}}> : tensor<4x2xf64>
  // CHECK: %[[PREPARED:.*]] = lattigo.ckks.prepare_linear_transform %{{.*}}, %{{.*}}, %[[DIAGONALS]] {diagonal_indices = array<i32: 0, 2>, levelQ = 0 : i64, logBabyStepGiantStepRatio = 0 : i64, logSlots = 9 : i64, source_row_indices = array<i32: 1, 3>} : ({{.*}}, tensor<4x2xf64>) -> !linear_transformation
  // CHECK: %[[OUT:.*]] = lattigo.ckks.apply_linear_transform %{{.*}}, %{{.*}}, %[[PREPARED]] : ({{.*}}) -> !ct
  // CHECK: return %[[OUT]] : !ct
  func.func @test_prepare_apply(%ct: !ct) -> !ct {
    %diagonals = arith.constant dense<[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]> : tensor<4x2xf64>
    %lt = kernel.prepare_linear_transform %diagonals {diagonal_indices = array<i64: 0, 2>, source_row_indices = array<i64: 1, 3>} : tensor<4x2xf64> -> !prepared
    %0 = kernel.apply_linear_transform %ct, %lt : !ct, !prepared -> !ct
    return %0 : !ct
  }
}
