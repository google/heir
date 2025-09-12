// RUN: heir-opt --polynomial-to-mod-arith="build-materializations=false" --split-input-file %s | FileCheck %s

!Z1032955396097_i64 = !mod_arith.int<1032955396097 : i64>
!Z1095233372161_i64 = !mod_arith.int<1095233372161 : i64>
!rns_2 = !rns.rns<!Z1095233372161_i64, !Z1032955396097_i64>
#ring_rns_2 = #polynomial.ring<coefficientType = !rns_2, polynomialModulus = <1 + x**1024>>
!poly = !polynomial.polynomial<ring = #ring_rns_2>

module attributes {
    ckks.schemeParam = #ckks.scheme_param<
      logN = 14,
      Q = [36028797019389953, 35184372121601, 35184372744193, 35184373006337, 35184373989377, 35184374874113],
      P = [36028797019488257, 36028797020209153],
      logDefaultScale = 45
    >
} {
  // the partition step has a single part that evenly divides the whole: 2 rns limbs and partSize=2
  // CHECK: @test_single_part
  // CHECK-SAME: (%[[poly:[^:]*]]: tensor<1024x!rns.rns
  // CHECK-SAME: , %[[ksk:[^:]*]]:
  func.func @test_single_part(%p: !poly, %ksk: tensor<10x2x!poly>) -> (!poly, !poly) {
    // CHECK: rns.extract_slice %[[poly]] {size = 2 : i32, start = 0 : i32}
    %c, %l = polynomial.key_switch_inner %p, %ksk : (!poly, tensor<10x2x!poly>) -> (!poly, !poly)
    return %c, %l : !poly, !poly
  }
}

// -----

// 11 primes, part size 3
!Z0 = !mod_arith.int<900015181768817533 : i64>
!Z1 = !mod_arith.int<143516525413762673 : i64>
!Z2 = !mod_arith.int<261405424692085787 : i64>
!Z3 = !mod_arith.int<820721655958272181 : i64>
!Z4 = !mod_arith.int<474168048327747811 : i64>
!Z5 = !mod_arith.int<997578949738158913 : i64>
!Z6 = !mod_arith.int<734673533440457291 : i64>
!Z7 = !mod_arith.int<783799894861165661 : i64>
!Z8 = !mod_arith.int<237657300033566549 : i64>
!Z9 = !mod_arith.int<480184796436462017 : i64>
!Z10 =!mod_arith.int<591169749652992061 : i64>

!rns_full = !rns.rns<!Z0, !Z1, !Z2, !Z3, !Z4, !Z5, !Z6, !Z7, !Z8, !Z9, !Z10>
#ring_full = #polynomial.ring<coefficientType = !rns_full, polynomialModulus = <1 + x**1024>>
!poly = !polynomial.polynomial<ring = #ring_full>

module attributes {
    ckks.schemeParam = #ckks.scheme_param<
      logN = 14,
      Q = [36028797019389953, 35184372121601, 35184372744193, 35184373006337, 35184373989377, 35184374874113],
      P = [36028797019488257, 383061492451512379, 479480638416718187],
      logDefaultScale = 45
    >
} {
  // CHECK: @test_indivisible_parts
  // CHECK-SAME: (%[[poly:[^:]*]]: tensor<1024x!rns.rns
  // CHECK-SAME: , %[[ksk:[^:]*]]:
  func.func @test_indivisible_parts(%p: !poly, %ksk: tensor<10x2x!poly>) -> (!poly, !poly) {
    // CHECK: rns.extract_slice %[[poly]] {size = 3 : i32, start = 0 : i32}
    // CHECK: rns.extract_slice %[[poly]] {size = 3 : i32, start = 3 : i32}
    // CHECK: rns.extract_slice %[[poly]] {size = 3 : i32, start = 6 : i32}
    // CHECK: rns.extract_slice %[[poly]] {size = 2 : i32, start = 9 : i32}
    %c, %l = polynomial.key_switch_inner %p, %ksk : (!poly, tensor<10x2x!poly>) -> (!poly, !poly)
    return %c, %l : !poly, !poly
  }
}
