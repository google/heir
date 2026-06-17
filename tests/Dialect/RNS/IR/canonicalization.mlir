// RUN: heir-opt -canonicalize %s | FileCheck %s

!Zp1 = !mod_arith.int<17 : i32>
!Zp2 = !mod_arith.int<19 : i32>
!rns = !rns.rns<!Zp1, !Zp2>

// CHECK: @test_rns_add_fold
// CHECK: () -> [[RNS_T:.*]] {
func.func @test_rns_add_fold() -> !rns {
  // CHECK: %[[RESULT:.+]] = rns.constant <[#mod_arith.value<12 : {{.+}}> : {{.+}}, #mod_arith.value<14 : {{.+}}> : {{.+}}]> : [[RNS_T]]
  %e1 = rns.constant <[#mod_arith.value<10 : !Zp1>, #mod_arith.value<5 : !Zp2>]> : !rns
  %e2 = rns.constant <[#mod_arith.value<2 : !Zp1>, #mod_arith.value<9 : !Zp2>]> : !rns
  %add = mod_arith.add %e1, %e2 : !rns
  // CHECK: return %[[RESULT]] : [[RNS_T]]
  return %add : !rns
}

// CHECK: @test_rns_sub_fold
// CHECK: () -> [[RNS_T:.*]] {
func.func @test_rns_sub_fold() -> !rns {
  // 10 - 2 mod 17 = 8. 5 - 9 mod 19 = 15.
  // CHECK: %[[RESULT:.+]] = rns.constant <[#mod_arith.value<8 : {{.+}}> : {{.+}}, #mod_arith.value<15 : {{.+}}> : {{.+}}]> : [[RNS_T]]
  %e1 = rns.constant <[#mod_arith.value<10 : !Zp1>, #mod_arith.value<5 : !Zp2>]> : !rns
  %e2 = rns.constant <[#mod_arith.value<2 : !Zp1>, #mod_arith.value<9 : !Zp2>]> : !rns
  %sub = mod_arith.sub %e1, %e2 : !rns
  // CHECK: return %[[RESULT]] : [[RNS_T]]
  return %sub : !rns
}

// CHECK: @test_rns_mul_fold
// CHECK: () -> [[RNS_T:.*]] {
func.func @test_rns_mul_fold() -> !rns {
  // 3 * 4 mod 17 = 12. 5 * 6 mod 19 = 11.
  // CHECK: %[[RESULT:.+]] = rns.constant <[#mod_arith.value<12 : {{.+}}> : {{.+}}, #mod_arith.value<11 : {{.+}}> : {{.+}}]> : [[RNS_T]]
  %e1 = rns.constant <[#mod_arith.value<3 : !Zp1>, #mod_arith.value<5 : !Zp2>]> : !rns
  %e2 = rns.constant <[#mod_arith.value<4 : !Zp1>, #mod_arith.value<6 : !Zp2>]> : !rns
  %mul = mod_arith.mul %e1, %e2 : !rns
  // CHECK: return %[[RESULT]] : [[RNS_T]]
  return %mul : !rns
}

// CHECK: @test_rns_mac_fold
// CHECK: () -> [[RNS_T:.*]] {
func.func @test_rns_mac_fold() -> !rns {
  // 3 * 4 + 2 mod 17 = 14. 5 * 6 + 1 mod 19 = 12.
  // CHECK: %[[RESULT:.+]] = rns.constant <[#mod_arith.value<14 : {{.+}}> : {{.+}}, #mod_arith.value<12 : {{.+}}> : {{.+}}]> : [[RNS_T]]
  %e1 = rns.constant <[#mod_arith.value<3 : !Zp1>, #mod_arith.value<5 : !Zp2>]> : !rns
  %e2 = rns.constant <[#mod_arith.value<4 : !Zp1>, #mod_arith.value<6 : !Zp2>]> : !rns
  %e3 = rns.constant <[#mod_arith.value<2 : !Zp1>, #mod_arith.value<1 : !Zp2>]> : !rns
  %mac = mod_arith.mac %e1, %e2, %e3 : !rns
  // CHECK: return %[[RESULT]] : [[RNS_T]]
  return %mac : !rns
}
