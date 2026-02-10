// RUN: heir-opt -canonicalize %s | FileCheck %s

#ntt_poly = #polynomial.int_polynomial<-1 + x**8>
!coeff_ty = !mod_arith.int<256:i32>
#ntt_ring = #polynomial.ring<coefficientType=!coeff_ty, polynomialModulus=#ntt_poly>
#root_val = #mod_arith.value<31:!coeff_ty>
#root = #polynomial.primitive_root<value=#root_val, degree=8:i32>
!poly_ty = !polynomial.polynomial<ring=#ntt_ring>
!ntt_poly_ty = !polynomial.polynomial<ring=#ntt_ring, form=<isCoeffForm=false>>
!tensor_ty = tensor<8x!coeff_ty, #ntt_ring>

// CHECK: @test_canonicalize_intt_after_ntt
// CHECK: (%[[P:.*]]: [[T:.*]]) -> [[T]]
func.func @test_canonicalize_intt_after_ntt(%p0 : !poly_ty) -> !poly_ty {
  // CHECK-NOT: polynomial.ntt
  // CHECK-NOT: polynomial.intt
  // CHECK: %[[RESULT:.+]] = polynomial.add %[[P]], %[[P]]  : [[T]]
  %t0 = polynomial.ntt %p0 {root=#root} : !poly_ty
  %p1 = polynomial.intt %t0 {root=#root} : !ntt_poly_ty
  %p2 = polynomial.add %p1, %p1 : !poly_ty
  // CHECK: return %[[RESULT]] : [[T]]
  return %p2 : !poly_ty
}

// CHECK: @test_canonicalize_ntt_after_intt
// CHECK: (%[[X:.*]]: [[T:.*]]) -> [[T]]
func.func @test_canonicalize_ntt_after_intt(%t0 : !ntt_poly_ty) -> !ntt_poly_ty {
  // CHECK-NOT: polynomial.intt
  // CHECK-NOT: polynomial.ntt
  // CHECK: %[[RESULT:.+]] = polynomial.add %[[X]], %[[X]] : [[T]]
  %p0 = polynomial.intt %t0 {root=#root} : !ntt_poly_ty
  %t1 = polynomial.ntt %p0 {root=#root} : !poly_ty
  %t2 = polynomial.add %t1, %t1 : !ntt_poly_ty
  // CHECK: return %[[RESULT]] : [[T]]
  return %t2 : !ntt_poly_ty
}
