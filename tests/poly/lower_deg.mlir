// RUN: heir-opt --poly-to-standard %s > %t
// RUN: FileCheck %s < %t

#cycl_2048 = #poly.polynomial<1 + x**1024>
#ring = #poly.ring<cmod=4294967296, ideal=#cycl_2048>

func.func @lower_poly_degree() {
  // 2 + 2x + 2x^2 + ... + 2x^{1023}
  // CHECK: [[X:%.+]] = arith.constant dense<2> : [[T:tensor<1024xi32>]]
  %coeffs1 = arith.constant dense<2> : tensor<1024xi32>
  // CHECK-NOT: poly.from_tensor
  %poly0 = poly.from_tensor %coeffs1 : tensor<1024xi32> -> !poly.poly<#ring>
  // CHECK-NOT: poly.degree
  // CHECK: %[[C1023:.+]] = arith.constant 1023 : index
  // CHECK: scf.while (%[[ARG0:.*]] = %[[C1023]]) : (index) -> index {
  // CHECK:    tensor.extract
  // CHECK:    arith.cmpi
  // CHECK:    scf.condition
  // CHECK: } do {
  // CHECK:  ^bb0
  // CHECK:    arith.subi
  // CHECK:    scf.yield
  // CHECK: }
  %0 = poly.degree %poly0 : !poly.poly<#ring>
  // CHECK: return
  return
}
