// RUN: heir-opt --polynomial-to-mod-arith --mlir-print-local-scope %s | FileCheck %s

#poly = #polynomial.int_polynomial<1 + x**1024>
!coeff_ty = !mod_arith.int<2837465:i32>
#ring = #polynomial.ring<coefficientType=!coeff_ty, polynomialModulus=#poly>

// CHECK: func @test_apply_coefficientwise
// CHECK-SAME: (%[[ARG0:.*]]: tensor<1024x!mod_arith.int<2837465 : i32>>)
func.func @test_apply_coefficientwise(%p0 : !polynomial.polynomial<ring=#ring>) -> !polynomial.polynomial<ring=#ring> {
  // CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<1024x!mod_arith.int<2837465 : i32>>
  // CHECK: %[[FOR:.*]] = scf.for %[[IV:.*]] = %c0 to %c1024 step %c1 iter_args(%[[ITER:.*]] = %[[EMPTY]]) -> (tensor<1024x!mod_arith.int<2837465 : i32>>) {
  // CHECK:   %[[COEFF:.*]] = tensor.extract %[[ARG0]][%[[IV]]]
  // CHECK:   %[[ADD:.*]] = mod_arith.add %[[COEFF]], %[[COEFF]]
  // CHECK:   %[[INSERT:.*]] = tensor.insert %[[ADD]] into %[[ITER]][%[[IV]]]
  // CHECK:   scf.yield %[[INSERT]]
  // CHECK: }
  // CHECK: return %[[FOR]]
  %1 = polynomial.apply_coefficientwise (%p0 : !polynomial.polynomial<ring=#ring>) {
  ^body(%coeff: !coeff_ty, %degree: index):
    %2 = mod_arith.add %coeff, %coeff : !coeff_ty
    polynomial.yield %2 : !coeff_ty
  } -> !polynomial.polynomial<ring=#ring>
  return %1 : !polynomial.polynomial<ring=#ring>
}

#ring2 = #polynomial.ring<coefficientType=i32, polynomialModulus=#poly>
// CHECK: func @test_apply_coefficientwise_int
// CHECK-SAME: (%[[ARG0:.*]]: tensor<1024xi32>)
func.func @test_apply_coefficientwise_int(%p0 : !polynomial.polynomial<ring=#ring2>) -> !polynomial.polynomial<ring=#ring2> {
  // CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<1024xi32>
  // CHECK: %[[FOR:.*]] = scf.for %[[IV:.*]] = %c0 to %c1024 step %c1 iter_args(%[[ITER:.*]] = %[[EMPTY]]) -> (tensor<1024xi32>) {
  // CHECK:   %[[COEFF:.*]] = tensor.extract %[[ARG0]][%[[IV]]]
  // CHECK:   %[[ADD:.*]] = arith.addi %[[COEFF]], %[[COEFF]]
  // CHECK:   %[[INSERT:.*]] = tensor.insert %[[ADD]] into %[[ITER]][%[[IV]]]
  // CHECK:   scf.yield %[[INSERT]]
  // CHECK: }
  // CHECK: return %[[FOR]]
  %1 = polynomial.apply_coefficientwise (%p0 : !polynomial.polynomial<ring=#ring2>) {
  ^body(%coeff: i32, %degree: index):
    %2 = arith.addi %coeff, %coeff : i32
    polynomial.yield %2 : i32
  } -> !polynomial.polynomial<ring=#ring2>
  return %1 : !polynomial.polynomial<ring=#ring2>
}
