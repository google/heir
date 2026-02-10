// RUN: heir-opt --convert-polynomial-mul-to-ntt %s | FileCheck %s

!Zq0 = !mod_arith.int<1095233372161 : i64>
!Zq1 = !mod_arith.int<1032955396097 : i64>
#ring_1 = #polynomial.ring<coefficientType = !rns.rns<!Zq0>, polynomialModulus = <1 + x**1024>>
!poly_ty_1 = !polynomial.polynomial<ring=#ring_1, form=coeff>

#ring_2 = #polynomial.ring<coefficientType = !rns.rns<!Zq0, !Zq1>, polynomialModulus = <1 + x**1024>>
!poly_ty_2 = !polynomial.polynomial<ring=#ring_2, form=coeff>

module {
  // Covers: a shared-dataflow circuit where optimal placement must avoid
  // redundant conversions on mul chains while still satisfying coeff-only
  // consumers. The optimal weighted conversion count here is 3:
  // 1 NTT on %x, and 2 INTTs (on %diff and %prod).
  // CHECK: func.func @optimal_ntt_placement(
  // CHECK-COUNT-1: polynomial.ntt
  // CHECK-COUNT-2: polynomial.intt
  func.func @optimal_ntt_placement(%x: !poly_ty_2, %k: index) -> !poly_ty_1 {
    %x_sq = polynomial.mul %x, %x : !poly_ty_2
    %x_cu = polynomial.mul %x_sq, %x : !poly_ty_2

    %sum = polynomial.add %x_sq, %x_cu : !poly_ty_2
    %diff = polynomial.sub %x_cu, %x_sq : !poly_ty_2

    // coeff-only consumer of %diff
    %shift = polynomial.monic_monomial_mul %diff, %k : (!poly_ty_2, index) -> !poly_ty_2

    // coeff-only consumer of %x, forcing both coeff/eval demand on the input
    %x_t = polynomial.to_tensor %x : !poly_ty_2 -> tensor<1024x!rns.rns<!Zq0, !Zq1>>
    %shift_t = polynomial.to_tensor %shift : !poly_ty_2 -> tensor<1024x!rns.rns<!Zq0, !Zq1>>

    %prod = polynomial.mul %sum, %sum : !poly_ty_2
    // coeff-only consumer of %prod
    %out = polynomial.convert_basis %prod {targetBasis = !rns.rns<!Zq0>} : !poly_ty_2
    return %out : !poly_ty_1
  }
}
