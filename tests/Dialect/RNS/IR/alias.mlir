// RUN: heir-opt %s | FileCheck %s

!Zp = !mod_arith.int<17 : i10>
!Zp1 = !mod_arith.int<17 : i32>
!Zp2 = !mod_arith.int<65536 : i32>
!Zp3 = !mod_arith.int<65537 : i32>

// CHECK: !rns_Z17_i32_Z65536_i32_ = !rns.rns<!Z17_i32_, !Z65536_i32_>
!rns = !rns.rns<!Zp1, !Zp2>
// CHECK: !rns_Z17_i32_Z65536_i32_Z65537_i32_ = !rns.rns<!Z17_i32_, !Z65536_i32_, !Z65537_i32_>
!rns1 = !rns.rns<!Zp1, !Zp2, !Zp3>

// CHECK-LABEL: @test_alias
func.func @test_alias(%0 : !rns, %1 : !rns1) -> !rns {
    return %0 : !rns
}
