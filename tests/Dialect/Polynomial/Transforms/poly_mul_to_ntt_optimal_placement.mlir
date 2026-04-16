// RUN: heir-opt --convert-polynomial-mul-to-ntt %s | FileCheck %s

!Zq0 = !mod_arith.int<1095233372161 : i64>
!Zq1 = !mod_arith.int<1032955396097 : i64>
!Zq2 = !mod_arith.int<1002198353921 : i64>
#ring_1 = #polynomial.ring<coefficientType = !rns.rns<!Zq0>, polynomialModulus = <1 + x**1024>>
!poly_ty_1 = !polynomial.polynomial<ring=#ring_1, form=coeff>
!ntt_poly_ty_1 = !polynomial.polynomial<ring=#ring_1, form=eval>

#ring_2 = #polynomial.ring<coefficientType = !rns.rns<!Zq0, !Zq1>, polynomialModulus = <1 + x**1024>>
!poly_ty_2 = !polynomial.polynomial<ring=#ring_2, form=coeff>

#ring_3 = #polynomial.ring<coefficientType = !rns.rns<!Zq0, !Zq1, !Zq2>, polynomialModulus = <1 + x**1024>>
!poly_ty_3 = !polynomial.polynomial<ring=#ring_3, form=coeff>

#ring_4 = #polynomial.ring<coefficientType = !rns.rns<!Zq1>, polynomialModulus = <1 + x**1024>>
!poly_ty_4 = !polynomial.polynomial<ring=#ring_4, form=coeff>
!ntt_poly_ty_4 = !polynomial.polynomial<ring=#ring_4, form=eval>

module {
  // Covers: a shared-dataflow circuit where optimal placement must avoid
  // redundant conversions on mul chains while still satisfying coeff-only
  // consumers. The optimal weighted conversion count here is 3:
  // 1 NTT on %x, and 2 INTTs (on %diff and %prod).
  // CHECK: func.func @optimal_ntt_placement(
  // CHECK-NOT: polynomial.intt
  // CHECK: polynomial.ntt
  // CHECK-NOT: polynomial.ntt
  // CHECK: polynomial.intt
  // CHECK-NOT: polynomial.ntt
  // CHECK: polynomial.intt
  // CHECK-NOT: polynomial.ntt
  // CHECK-NOT: polynomial.intt
  // CHECK: return
  func.func @optimal_ntt_placement(%x: !poly_ty_2, %k: index) -> (!poly_ty_1, tensor<1024x!rns.rns<!Zq0, !Zq1>>, tensor<1024x!rns.rns<!Zq0, !Zq1>>) {
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
    %out = polynomial.convert_basis %prod {targetBasis = !rns.rns<!Zq0>} : !poly_ty_2 -> !poly_ty_1
    return %out, %x_t, %shift_t : !poly_ty_1, tensor<1024x!rns.rns<!Zq0, !Zq1>>, tensor<1024x!rns.rns<!Zq0, !Zq1>>
  }

  // Covers: limb-aware conversion costs can move NTTs past extract_slice.
  // Old unweighted costs would hoist one NTT on %x; the weighted model keeps
  // %x in coeff form and inserts two cheaper one-limb NTTs on the slices.
  // CHECK: func.func @weighted_extract_slice_ntt_placement([[x:%.+]]: [[poly_ty_3:![^ ]+]]) -> ([[ntt_poly_ty_1:![^ ]+]], [[ntt_poly_ty_4:![^ ]+]], tensor<1024x[[RNS3:![^ ]+]]>) {
  // CHECK-NOT: polynomial.ntt
  // CHECK: [[a:%.+]] = polynomial.extract_slice [[x]] {size = 1 : index, start = 0 : index} : [[poly_ty_3]] -> [[poly_ty_1:![^ ]+]]
  // CHECK: [[a_ntt:%.+]] = polynomial.ntt [[a]] : [[poly_ty_1]]
  // CHECK: [[b:%.+]] = polynomial.extract_slice [[x]] {size = 1 : index, start = 1 : index} : [[poly_ty_3]] -> [[poly_ty_4:![^ ]+]]
  // CHECK: [[b_ntt:%.+]] = polynomial.ntt [[b]] : [[poly_ty_4]]
  // CHECK: [[ma:%.+]] = polynomial.mul [[a_ntt]], [[a_ntt]] : [[ntt_poly_ty_1]]
  // CHECK: [[mb:%.+]] = polynomial.mul [[b_ntt]], [[b_ntt]] : [[ntt_poly_ty_4]]
  // CHECK: [[x_t:%.+]] = polynomial.to_tensor [[x]] : [[poly_ty_3]] -> tensor<1024x[[RNS3]]>
  // CHECK: return [[ma]], [[mb]], [[x_t]] : [[ntt_poly_ty_1]], [[ntt_poly_ty_4]], tensor<1024x[[RNS3]]>
  func.func @weighted_extract_slice_ntt_placement(%x: !poly_ty_3) -> (!poly_ty_1, !poly_ty_4, tensor<1024x!rns.rns<!Zq0, !Zq1, !Zq2>>) {
    %a = polynomial.extract_slice %x {start = 0 : index, size = 1 : index}
        : !poly_ty_3 -> !poly_ty_1
    %b = polynomial.extract_slice %x {start = 1 : index, size = 1 : index}
        : !poly_ty_3 -> !poly_ty_4
    %ma = polynomial.mul %a, %a : !poly_ty_1
    %mb = polynomial.mul %b, %b : !poly_ty_4
    %x_t = polynomial.to_tensor %x : !poly_ty_3 -> tensor<1024x!rns.rns<!Zq0, !Zq1, !Zq2>>
    return %ma, %mb, %x_t : !poly_ty_1, !poly_ty_4, tensor<1024x!rns.rns<!Zq0, !Zq1, !Zq2>>
  }
}
