// RUN: heir-opt --convert-polynomial-mul-to-ntt %s | FileCheck %s

// CHECK-DAG: [[ZQ0:![^ ]+]] = !mod_arith.int<1095233372161 : i64>
!Zq0 = !mod_arith.int<1095233372161 : i64>
// CHECK-DAG: [[ZQ1:![^ ]+]] = !mod_arith.int<1032955396097 : i64>
!Zq1 = !mod_arith.int<1032955396097 : i64>
// CHECK-DAG: [[RNS2:![^ ]+]] = !rns.rns<[[ZQ0]], [[ZQ1]]>
// CHECK-DAG: [[ring_2:#[^ ]+]] = #polynomial.ring<coefficientType = [[RNS2]], polynomialModulus = <1 + x**1024>>
#ring_2 = #polynomial.ring<coefficientType = !rns.rns<!Zq0, !Zq1>, polynomialModulus = <1 + x**1024>>
// CHECK-DAG: [[poly_ty_2:![^ ]+]] = !polynomial.polynomial<ring = [[ring_2]]>
!poly_ty_2 = !polynomial.polynomial<ring=#ring_2, form=coeff>


// Covers: flexible-op-only pipeline (Add/Sub) without mandatory coeff/eval-only consumers.
// CHECK: func.func @test_ntt_insertion5([[x5:%.+]]: [[poly_ty_2]]) -> [[poly_ty_2]] {
// CHECK: [[a5:%.+]] = polynomial.add [[x5]], [[x5]] : [[poly_ty_2]]
// CHECK: [[b5:%.+]] = polynomial.sub [[a5]], [[x5]] : [[poly_ty_2]]
// CHECK: return [[b5]] : [[poly_ty_2]]
func.func @test_ntt_insertion5(%x: !poly_ty_2) -> !poly_ty_2 {
  %a = polynomial.add %x, %x : !poly_ty_2
  %b = polynomial.sub %a, %x : !poly_ty_2
  return %b : !poly_ty_2
}


