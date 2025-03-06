// RUN: heir-opt %s | FileCheck %s

// CHECK: !Z17_i10 = !mod_arith.int<17 : i10>
!Zp = !mod_arith.int<17 : i10>
// CHECK: !Z17_i32 = !mod_arith.int<17 : i32>
!Zp1 = !mod_arith.int<17 : i32>
// CHECK: !Z65536_i32 = !mod_arith.int<65536 : i32>
!Zp2 = !mod_arith.int<65536 : i32>

// CHECK-LABEL: @test_alias
func.func @test_alias(%0 : !Zp, %1 : !Zp1, %2 : !Zp2) -> !Zp {
    return %0 : !Zp
}
