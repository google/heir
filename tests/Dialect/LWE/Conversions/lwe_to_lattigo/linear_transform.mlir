// RUN: heir-opt --lwe-to-lattigo %s | FileCheck %s

#inverse_canonical_encoding = #lwe.inverse_canonical_encoding<scaling_factor = 45>
#key = #lwe.key<>
#modulus_chain = #lwe.modulus_chain<elements = <36028797018652673 : i64, 35184372121601 : i64>, current = 0>
#ring_f64_1_x1024 = #polynomial.ring<coefficientType = f64, polynomialModulus = <1 + x**1024>>
!rns_L0 = !rns.rns<!mod_arith.int<36028797018652673 : i64>>
#ring_rns_L0_1_x1024 = #polynomial.ring<coefficientType = !rns_L0, polynomialModulus = <1 + x**1024>>
#ciphertext_space_L0 = #lwe.ciphertext_space<ring = #ring_rns_L0_1_x1024, encryption_type = mix>
!ct = !lwe.lwe_ciphertext<plaintext_space = <ring = #ring_f64_1_x1024, encoding = #inverse_canonical_encoding>, ciphertext_space = #ciphertext_space_L0, key = #key, modulus_chain = #modulus_chain>

// CHECK: ![[CT:.*]] = !lattigo.rlwe.ciphertext
// CHECK: ![[ENCODER:.*]] = !lattigo.ckks.encoder
// CHECK: ![[EVAL:.*]] = !lattigo.ckks.evaluator

module attributes {backend.lattigo, ckks.schemeParam = #ckks.scheme_param<logN = 13, Q = [36028797018652673, 35184372121601], P = [1152921504606994433], logDefaultScale = 45, encryptionTechnique = extended>, scheme.ckks} {
  // CHECK: func.func @test_linear_transform(%[[EVAL:.*]]: ![[EVAL]], %{{.*}}: {{.*}}, %[[ENCODER:.*]]: ![[ENCODER]], %[[CT:.*]]: ![[CT]]{{.*}}) -> ![[CT]]
  // CHECK: %[[DIAGONALS:.*]] = arith.constant dense<{{\[\[}}1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00{{\]\]}}> : tensor<2x2xf64>
  // CHECK: %[[VAL_3:.*]] = lattigo.ckks.linear_transform %[[EVAL]], %[[ENCODER]], %[[CT]], %[[DIAGONALS]] {diagonal_indices = array<i32: 0, 1>, levelQ = 1 : i64, logBabyStepGiantStepRatio = 0 : i64} : (![[EVAL]], ![[ENCODER]], ![[CT]], tensor<2x2xf64>) -> ![[CT]]
  // CHECK: return %[[VAL_3]] : ![[CT]]
  func.func @test_linear_transform(%ct: !ct) -> !ct {
    %0 = kernel.linear_transform %ct {diagonals = dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf64>, diagonal_indices = array<i64: 0, 1>} : !ct -> !ct
    return %0 : !ct
  }
}
