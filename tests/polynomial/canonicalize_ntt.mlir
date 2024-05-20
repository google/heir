// RUN: heir-opt --canonicalize %s | FileCheck %s

#cycl_1024 = #_polynomial.polynomial<1 + x**1024>
#ring = #_polynomial.ring<cmod=3758161921, ideal=#cycl_1024, root=376008217>

!tensor_ty = tensor<1024xi32, #ring>
!poly_ty = !_polynomial.polynomial<#ring>

// CHECK: module
module {
  // CHECK-LABEL: @test_canonicalize_intt_after_ntt
  // CHECK: (%[[P:.*]]: [[T:.*]]) -> [[T]]
  func.func @test_canonicalize_intt_after_ntt(%p0 : !poly_ty) -> !poly_ty {
    // CHECK-NOT: _polynomial.ntt
    // CHECK-NOT: _polynomial.intt
    // CHECK: %[[RESULT:.+]] = _polynomial.add(%[[P]], %[[P]]) : [[T]]
    %t0 = _polynomial.ntt %p0 : !poly_ty -> !tensor_ty
    %p1 = _polynomial.intt %t0: !tensor_ty -> !poly_ty
    %p2 = _polynomial.add(%p1, %p1): !poly_ty
    // CHECK: return %[[RESULT]] : [[T]]
    return %p2 : !poly_ty
  }

  // CHECK-LABEL: @test_canonicalize_ntt_after_intt
  // CHECK: (%[[X:.*]]: [[T:.*]]) -> [[T]]
  func.func @test_canonicalize_ntt_after_intt(%t0 : !tensor_ty) -> !tensor_ty {
    // CHECK-NOT: _polynomial.intt
    // CHECK-NOT: _polynomial.ntt
    // CHECK: %[[RESULT:.+]] = arith.addi %[[X]], %[[X]] : [[T]]
    %p0 = _polynomial.intt %t0 : !tensor_ty -> !poly_ty
    %t1 = _polynomial.ntt %p0: !poly_ty -> !tensor_ty
    %t2 = arith.addi %t1, %t1: !tensor_ty
    // CHECK: return %[[RESULT]] : [[T]]
    return %t2 : !tensor_ty
  }
}
