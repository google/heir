// RUN: heir-opt --mlir-print-local-scope --convert-elementwise-to-affine %s | FileCheck %s

#encoding = #lwe.full_crt_packing_encoding<scaling_factor = 0>
#poly_1024 = #polynomial.int_polynomial<1 + x**1024>
!plaintext_coefficient_modulus = !mod_arith.int<65537 : i64>
#plaintext_ring = #polynomial.ring<
  coefficientType=!plaintext_coefficient_modulus,
  polynomialModulus=#poly_1024>
#plaintext_space = #lwe.plaintext_space<
  ring=#plaintext_ring,
  encoding=#encoding>

!pt = !lwe.lwe_plaintext<
  plaintext_space=#plaintext_space>

// CHECK: @test_rlwe_encode_elementwise
// CHECK-SAME: (%[[ARG0:.*]]: tensor<4x1024xi16>) -> tensor<4x!lwe.lwe_plaintext
func.func @test_rlwe_encode_elementwise(%arg0: tensor<4x1024xi16>) -> tensor<4x!pt> {
  // CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<4x!lwe.lwe_plaintext
  // CHECK: %[[LOOP:.*]] = affine.for %[[I:.*]] = 0 to 4 iter_args(%[[T0:.*]] = %[[EMPTY]]) -> (tensor<4x!lwe.lwe_plaintext
  // CHECK:   %[[SLICE:.*]] = tensor.extract_slice %[[ARG0]][%[[I]], 0] [1, 1024] [1, 1] : tensor<4x1024xi16> to tensor<1024xi16>
  // CHECK:   %[[ENCODED:.*]] = lwe.rlwe_encode %[[SLICE]] {encoding = #lwe.full_crt_packing_encoding<scaling_factor = 0>, ring = #polynomial.ring<coefficientType = !mod_arith.int<65537 : i64>, polynomialModulus = <1 + x**1024>>} : tensor<1024xi16> -> !lwe.lwe_plaintext
  // CHECK:   %[[INSERTED:.*]] = tensor.insert %[[ENCODED]] into %[[T0]][%[[I]]] : tensor<4x!lwe.lwe_plaintext
  // CHECK:   affine.yield %[[INSERTED]] : tensor<4x!lwe.lwe_plaintext
  // CHECK: return %[[LOOP]] : tensor<4x!lwe.lwe_plaintext
  %0 = lwe.rlwe_encode %arg0 {encoding = #encoding, ring = #plaintext_ring} : tensor<4x1024xi16> -> tensor<4x!pt>
  return %0 : tensor<4x!pt>
}
