// RUN: heir-opt --polynomial-to-standard %s > %t
// RUN: FileCheck %s < %t

#cycl_2048 = #_polynomial.polynomial<1 + x**1024>
#ring = #_polynomial.ring<cmod=4294967296, ideal=#cycl_2048>

func.func @lower_leading_term() {
  // 2 + 2x + 2x^2 + ... + 2x^{1023}
  // CHECK: %[[X:.+]] = arith.constant dense<2> : [[T:tensor<1024xi32>]]
  // CHECK: %[[C0:.*]] = arith.constant 0 : i32
  // CHECK: %[[C1:.*]] = arith.constant 1 : index
  %coeffs1 = arith.constant dense<2> : tensor<1024xi32>
  // CHECK-NOT: _polynomial.from_tensor
  %poly0 = _polynomial.from_tensor %coeffs1 : tensor<1024xi32> -> !_polynomial.polynomial<#ring>
  // CHECK-NOT: _polynomial.leading_term
  // CHECK: %[[C1023:.+]] = arith.constant 1023 : index
  // CHECK: %[[WHILE_RES:.*]] = scf.while (%[[ARG0:.*]] = %[[C1023]]) : (index) -> index {
  // CHECK:    %[[EXTRACTED:.*]] = tensor.extract %[[X]][%[[ARG0]]] : [[T]]
  // CHECK:    %[[CMP:.*]] = arith.cmpi eq, %[[EXTRACTED]], %[[C0]]
  // CHECK:    scf.condition(%[[CMP]]) %[[ARG0]] : index
  // CHECK: } do {
  // CHECK:  ^bb0(%[[BODY_ARG:.*]]: index):
  // CHECK:    %[[SUBBED:.*]] = arith.subi %[[BODY_ARG]], %[[C1]]
  // CHECK:    scf.yield %[[SUBBED]] : index
  // CHECK: }
  // CHECK: tensor.extract %[[X]][%[[WHILE_RES]]] : [[T]]
  %0, %1 = _polynomial.leading_term %poly0 : !_polynomial.polynomial<#ring> -> (index, i32)
  // CHECK: return
  return
}
