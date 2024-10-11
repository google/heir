// RUN: heir-opt %s --lwe-to-polynomial | FileCheck %s

#encoding = #lwe.polynomial_coefficient_encoding<cleartext_start=15, cleartext_bitwidth=4>
#my_poly = #polynomial.int_polynomial<1 + x**1024>
#ring = #polynomial.ring<coefficientType = i32, coefficientModulus = 7917 : i32, polynomialModulus=#my_poly>
#rlwe_params = #lwe.rlwe_params<dimension=2, ring=#ring>
!plaintext_rlwe = !lwe.rlwe_plaintext<encoding = #encoding, ring = #ring, underlying_type=i3>
!ciphertext_rlwe = !lwe.rlwe_ciphertext<encoding = #encoding, rlwe_params = #rlwe_params, underlying_type=i3>
!rlwe_key = !lwe.rlwe_secret_key<rlwe_params=#rlwe_params>

func.func @test_rlwe_decrypt(%arg0: !ciphertext_rlwe, %arg1: !rlwe_key) -> !plaintext_rlwe {
  // CHECK-NOT: lwe.rlwe_decrypt

  // CHECK-DAG: %[[ZERO:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[ONE:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[CTXT_ZERO:.*]] = tensor.extract %arg0[%[[ZERO]]]
  // CHECK-DAG: %[[CTXT_ONE:.*]] = tensor.extract %arg0[%[[ONE]]]
  // CHECK-DAG: %[[SECRET_KEY:.*]] = tensor.extract %arg1[%[[ZERO]]]

  // CHECK:     %[[KEY_TIMES_CTXT_ZERO:.*]] = polynomial.mul %[[SECRET_KEY]], %[[CTXT_ZERO]]
  // CHECK:     %[[DECRYPTED_PTXT:.*]] = polynomial.add %[[KEY_TIMES_CTXT_ZERO]], %[[CTXT_ONE]]
  // CHECK:     return %[[DECRYPTED_PTXT]]
  %0 = lwe.rlwe_decrypt %arg0, %arg1 : (!ciphertext_rlwe, !rlwe_key) -> !plaintext_rlwe
  return %0 : !plaintext_rlwe
}
