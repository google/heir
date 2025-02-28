// RUN: heir-opt --apply-folders --split-input-file %s | FileCheck %s

!Zp = !mod_arith.int<255 : i32>

// CHECK-LABEL: @test_add_zero
// CHECK-SAME: (%[[ARG0:.*]]: !
func.func @test_add_zero(%arg0 : !Zp) -> !Zp {
  // CHECK: return %[[ARG0]]
  %c0 = mod_arith.constant 0 : !Zp
  %add = mod_arith.add %arg0, %c0 : !Zp
  return %add : !Zp
}

// -----

!Zp = !mod_arith.int<255 : i32>

// CHECK-LABEL: @test_add_sub_a_b_b
// CHECK-SAME: (%[[A:.*]]: ![[type:.*]], %[[B:.*]]: ![[type]])
func.func @test_add_sub_a_b_b(%a : !Zp, %b : !Zp) -> !Zp {
  // CHECK: return %[[A:.*]]
  %sub = mod_arith.sub %a, %b : !Zp
  %add = mod_arith.add %sub, %b : !Zp
  return %add : !Zp
}

// -----

!Zp = !mod_arith.int<255 : i32>

// CHECK-LABEL: @test_add_b_sub_a_b
// CHECK-SAME: (%[[A:.*]]: ![[type:.*]], %[[B:.*]]: ![[type]])
func.func @test_add_b_sub_a_b(%a : !Zp, %b : !Zp) -> !Zp {
  // CHECK: return %[[A:.*]]
  %sub = mod_arith.sub %a, %b : !Zp
  %add = mod_arith.add %b, %sub : !Zp
  return %add : !Zp
}
