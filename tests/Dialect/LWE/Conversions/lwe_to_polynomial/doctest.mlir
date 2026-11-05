// RUN: heir-opt --lwe-to-polynomial %s | FileCheck %s

!Zq = !mod_arith.int<1095233372161 : i64>
!Zt = !mod_arith.int<65537 : i64>
#key = #lwe.key<>
#ring_Zt = #polynomial.ring<coefficientType = !Zt, polynomialModulus = <1 + x**8>>
#ring_Q = #polynomial.ring<coefficientType = !rns.rns<!Zq>, polynomialModulus = <1 + x**8>>
#plaintext_space = #lwe.plaintext_space<ring = #ring_Zt, encoding = #lwe.full_crt_packing_encoding<scaling_factor = 0>>
!ct = !lwe.lwe_ciphertext<plaintext_space = #plaintext_space, ciphertext_space = <ring = #ring_Q, encryption_type = lsb>, key = #key>
!ct_D3 = !lwe.lwe_ciphertext<plaintext_space = #plaintext_space, ciphertext_space = <ring = #ring_Q, encryption_type = lsb, size = 3>, key = #key>

// A ciphertext made of N polynomials is lowered to a tensor of N polynomials,
// and ciphertext arithmetic to polynomial arithmetic on its components.

// CHECK: func.func @add_and_mul(
// CHECK-SAME: %[[X:.*]]: tensor<2x![[POLY:.*]]>, %[[Y:.*]]: tensor<2x![[POLY]]>
// CHECK-SAME: -> tensor<3x![[POLY]]>
func.func @add_and_mul(%x: !ct, %y: !ct) -> !ct_D3 {
  // CHECK: %[[SUM:.*]] = polynomial.add %[[X]], %[[Y]] : tensor<2x![[POLY]]>
  %sum = lwe.radd %x, %y : (!ct, !ct) -> !ct

  // (s0, s1) * (y0, y1) = (s0 * y0, s0 * y1 + s1 * y0, s1 * y1)
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[S0:.*]] = tensor.extract %[[SUM]][%[[C0]]]
  // CHECK-DAG: %[[S1:.*]] = tensor.extract %[[SUM]][%[[C1]]]
  // CHECK-DAG: %[[Y0:.*]] = tensor.extract %[[Y]][%[[C0]]]
  // CHECK-DAG: %[[Y1:.*]] = tensor.extract %[[Y]][%[[C1]]]
  // CHECK-DAG: %[[Z0:.*]] = polynomial.mul %[[S0]], %[[Y0]]
  // CHECK-DAG: %[[S0Y1:.*]] = polynomial.mul %[[S0]], %[[Y1]]
  // CHECK-DAG: %[[S1Y0:.*]] = polynomial.mul %[[S1]], %[[Y0]]
  // CHECK-DAG: %[[Z1:.*]] = polynomial.add %[[S0Y1]], %[[S1Y0]]
  // CHECK-DAG: %[[Z2:.*]] = polynomial.mul %[[S1]], %[[Y1]]
  // CHECK: %[[Z:.*]] = tensor.from_elements %[[Z0]], %[[Z1]], %[[Z2]]
  %prod = lwe.rmul %sum, %y : (!ct, !ct) -> !ct_D3

  // CHECK: return %[[Z]]
  return %prod : !ct_D3
}
