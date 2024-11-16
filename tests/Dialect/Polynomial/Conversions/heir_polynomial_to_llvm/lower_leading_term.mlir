// RUN: heir-opt --polynomial-to-mod-arith %s | FileCheck %s

#cycl_2048 = #polynomial.int_polynomial<1 + x**1024>
!coeff_ty = !mod_arith.int<4294967296:i64>
#ring = #polynomial.ring<coefficientType=!coeff_ty, polynomialModulus=#cycl_2048>

func.func @lower_leading_term() -> !coeff_ty {
  // 2 + 2x + 2x^2 + ... + 2x^{1023}
  // CHECK: %[[Xraw:.+]] = arith.constant dense<2> : [[T:tensor<1024xi64>]]
  // CHECK: %[[X:.+]] = mod_arith.encapsulate %[[Xraw]] : [[T]] -> [[Tmod:.*]]
  // CHECK: %[[C0:.*]] = arith.constant 0 : i64
  // CHECK: %[[C1:.*]] = arith.constant 1 : index
  %coeffs1_ = arith.constant dense<2> : tensor<1024xi64>
  %coeffs1 = mod_arith.encapsulate %coeffs1_ : tensor<1024xi64> -> tensor<1024x!coeff_ty>
  // CHECK-NOT: polynomial.from_tensor
  %poly0 = polynomial.from_tensor %coeffs1 : tensor<1024x!coeff_ty> -> !polynomial.polynomial<ring=#ring>
  // CHECK-NOT: polynomial.leading_term
  // CHECK: %[[C1023:.+]] = arith.constant 1023 : index
  // CHECK: %[[WHILE_RES:.*]] = scf.while (%[[ARG0:.*]] = %[[C1023]]) : (index) -> index {
  // CHECK:    %[[EXTRACTED:.*]] = tensor.extract %[[X]][%[[ARG0]]] : [[Tmod]]
  // CHECK:    %[[REDUCED:.*]] = mod_arith.reduce %[[EXTRACTED]]
  // CHECK:    %[[REDUCED_EXTRACTED:.*]] = mod_arith.extract %[[REDUCED]]
  // CHECK:    %[[CMP:.*]] = arith.cmpi eq, %[[REDUCED_EXTRACTED]], %[[C0]]
  // CHECK:    scf.condition(%[[CMP]]) %[[ARG0]] : index
  // CHECK: } do {
  // CHECK:  ^bb0(%[[BODY_ARG:.*]]: index):
  // CHECK:    %[[SUBBED:.*]] = arith.subi %[[BODY_ARG]], %[[C1]]
  // CHECK:    scf.yield %[[SUBBED]] : index
  // CHECK: }
  // CHECK: tensor.extract %[[X]][%[[WHILE_RES]]] : [[Tmod]]
  %0, %1 = polynomial.leading_term %poly0 : !polynomial.polynomial<ring=#ring> -> (index, !coeff_ty)
  // CHECK: return
  return %1 : !coeff_ty
}
