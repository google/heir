// RUN: heir-opt --bgv-to-openfhe %s | FileCheck %s

#encoding = #lwe.polynomial_evaluation_encoding<cleartext_start = 16, cleartext_bitwidth = 16>
#params = #lwe.rlwe_params<ring = <coefficientType = i32, coefficientModulus = 463187969 : i32, polynomialModulus=#polynomial.int_polynomial<1 + x**32>>>
!ty1 = !lwe.rlwe_ciphertext<encoding = #encoding, rlwe_params = #params, underlying_type = tensor<32xi16>>
!ty2 = !lwe.rlwe_ciphertext<encoding = #encoding, rlwe_params = #params, underlying_type = i16>

// CHECK-LABEL: @test_lower_extract
// CHECK-SAME: %[[arg0:.*]]:
func.func @test_lower_extract(%arg0: !ty1) -> !ty2 {
  // CHECK: arith.constant 4 : index
  // CHECK: arith.constant dense<[0, 0, 0, 0, 1, [[unused:[0 ,]*]]]> : tensor<32xi16>
  // CHECK: lwe.rlwe_encode
  // CHECK: openfhe.mul_plain
  // CHECK: openfhe.rot
  // CHECK: lwe.reinterpret_underlying_type
  // CHECK: return
  %c4 = arith.constant 4 : index
  %0 = bgv.extract %arg0, %c4 : (!ty1, index) -> !ty2
  return %0 : !ty2
}
