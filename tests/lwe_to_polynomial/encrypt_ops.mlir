// RUN: heir-opt %s --lwe-to-polynomial | FileCheck %s

#encoding = #lwe.polynomial_coefficient_encoding<cleartext_start=15, cleartext_bitwidth=15>
#my_poly = #polynomial.int_polynomial<1 + x**1024>
#ring = #polynomial.ring<coefficientType = i32, coefficientModulus = 7917 : i32, polynomialModulus=#my_poly>
#rlwe_params = #lwe.rlwe_params<dimension=2, ring=#ring>
!plaintext_rlwe = !lwe.rlwe_plaintext<encoding = #encoding, ring = #ring, underlying_type=i3>
!ciphertext_rlwe = !lwe.rlwe_ciphertext<encoding = #encoding, rlwe_params = #rlwe_params, underlying_type=i3>
!rlwe_key = !lwe.rlwe_public_key<rlwe_params=#rlwe_params>

func.func @test_rlwe_encrypt(%arg0: !plaintext_rlwe, %arg1: !rlwe_key) -> !ciphertext_rlwe {
  // CHECK-NOT: lwe.rlwe_encrypt

  // CHECK-DAG: %[[ZERO:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[ONE:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[CONSTANT_T:.*]] = arith.constant 32768

  // CHECK-DAG: %[[PRNG:.*]] = random.init_prng %[[SEED:.*]]
  // CHECK-DAG: %[[UNIFORM:.*]] = random.discrete_uniform_distribution %[[PRNG]]
  // CHECK-DAG: %[[GAUSSIAN:.*]] = random.discrete_gaussian_distribution %[[PRNG]]

  // CHECK-DAG: %[[PK_0:.*]] = tensor.extract %arg1[%[[ZERO]]]
  // CHECK-DAG: %[[PK_1:.*]] = tensor.extract %arg1[%[[ONE]]]

  // CHECK-DAG: %[[SAMPLE_U:.*]] = random.sample %[[UNIFORM]]
  // CHECK-DAG: %[[U:.*]] = polynomial.from_tensor %[[SAMPLE_U]]
  // CHECK:     %[[SAMPLE_E_0:.*]] = random.sample %[[GAUSSIAN]]
  // CHECK:     %[[E_0:.*]] = polynomial.from_tensor %[[SAMPLE_E_0]]
  // CHECK:     %[[SAMPLE_E_1:.*]] = random.sample %[[GAUSSIAN]]
  // CHECK:     %[[E_1:.*]] = polynomial.from_tensor %[[SAMPLE_E_1]]

  // CHECK-DAG: %[[PK_0_TIMES_U:.*]] = polynomial.mul %[[PK_0]], %[[U]]
  // CHECK-DAG: %[[E_0_TIMES_T:.*]] = polynomial.mul_scalar %[[E_0]], %[[CONSTANT_T]]
  // CHECK-DAG: %[[PK_0_TIMES_U_PLUS_E_0_TIMES_T:.*]] = polynomial.add %[[PK_0_TIMES_U]], %[[E_0_TIMES_T]]
  // CHECK-DAG: %[[C_0:.*]] = polynomial.add %[[PK_0_TIMES_U_PLUS_E_0_TIMES_T]], %arg0

  // CHECK-DAG: %[[PK_1_TIMES_U:.*]] = polynomial.mul %[[PK_1]], %[[U]]
  // CHECK-DAG: %[[E_1_TIMES_T:.*]] = polynomial.mul_scalar %[[E_1]], %[[CONSTANT_T]]
  // CHECK-DAG: %[[C_1:.*]] = polynomial.add %[[PK_1_TIMES_U]], %[[E_1_TIMES_T]]

  // CHECK:     %[[C:.*]] = tensor.from_elements %[[C_0]], %[[C_1]]
  // CHECK:     return %[[C]]
  %0 = lwe.rlwe_encrypt %arg0, %arg1 : (!plaintext_rlwe, !rlwe_key) -> !ciphertext_rlwe
  return %0 : !ciphertext_rlwe
}
