// RUN: heir-opt --polynomial-to-mod-arith %s | FileCheck %s

!Z1032955396097_i64 = !mod_arith.int<1032955396097 : i64>
!Z1095233372161_i64 = !mod_arith.int<1095233372161 : i64>
!rns_L1 = !rns.rns<!Z1095233372161_i64, !Z1032955396097_i64>
!rns_new = !rns.rns<!Z1095233372161_i64>
#ring_rns_L1_1_x1024 = #polynomial.ring<coefficientType = !rns_L1, polynomialModulus = <1 + x**1024>>
#ring_new = #polynomial.ring<coefficientType = !rns_new, polynomialModulus = <1 + x**1024>>
!poly = !polynomial.polynomial<ring = #ring_rns_L1_1_x1024>
!poly_new = !polynomial.polynomial<ring = #ring_new>

module attributes {
    ckks.schemeParam = #ckks.scheme_param<
      logN = 14,
      Q = [36028797019389953, 35184372121601, 35184372744193, 35184373006337, 35184373989377, 35184374874113],
      P = [36028797019488257, 36028797020209153],
      logDefaultScale = 45
    >
} {
  // CHECK: @test_keyswitch_inner_lowering
  func.func @test_keyswitch_inner_lowering(%p: !poly, %ksk: tensor<10x2x!poly>) -> (!poly, !poly) {
    // CHECK-NOT: polynomial.key_switch_inner
    %c, %l = polynomial.key_switch_inner %p, %ksk : (!poly, tensor<10x2x!poly>) -> (!poly, !poly)
    return %c, %l : !poly, !poly
  }
}
