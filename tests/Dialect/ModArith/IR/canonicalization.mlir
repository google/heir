// RUN: heir-opt -canonicalize %s | FileCheck %s

!Zp = !mod_arith.int<42 : i64>

// CHECK-LABEL: @test_add_fold
// CHECK: () -> [[T:.*]] {
func.func @test_add_fold() -> !Zp {
  // CHECK: %[[RESULT:.+]] = mod_arith.constant 18 : [[T]]
  %e1 = mod_arith.constant 12 : !Zp
  %e2 = mod_arith.constant 34 : !Zp
  %add = mod_arith.add %e1, %e2 : !Zp
  %e3 = mod_arith.constant 56 : !Zp
  %add2 = mod_arith.add %add, %e3 : !Zp
  // CHECK: return %[[RESULT]] : [[T]]
  return %add2 : !Zp
}

// CHECK-LABEL: @test_sub_fold
// CHECK: () -> [[T:.*]] {
func.func @test_sub_fold() -> !Zp {
  // CHECK: %[[RESULT:.+]] = mod_arith.constant 6 : [[T]]
  %e1 = mod_arith.constant 12 : !Zp
  %e2 = mod_arith.constant 34 : !Zp
  %sub = mod_arith.sub %e1, %e2 : !Zp
  %e3 = mod_arith.constant 56 : !Zp
  %sub2 = mod_arith.sub %sub, %e3 : !Zp
  // CHECK: return %[[RESULT]] : [[T]]
  return %sub2 : !Zp
}

// CHECK-LABEL: @test_mul_fold
// CHECK: () -> [[T:.*]] {
func.func @test_mul_fold() -> !Zp {
  // CHECK: %[[RESULT:.+]] = mod_arith.constant 0 : [[T]]
  %e1 = mod_arith.constant 12 : !Zp
  %e2 = mod_arith.constant 34 : !Zp
  %mul = mod_arith.mul %e1, %e2 : !Zp
  %e3 = mod_arith.constant 56 : !Zp
  %mul2 = mod_arith.mul %mul, %e3 : !Zp
  return %mul2 : !Zp
}

// CHECK-LABEL: @test_add_zero_rhs
// CHECK: (%[[arg0:.*]]: [[T:.*]]) -> [[T]]
func.func @test_add_zero_rhs(%x: !Zp) -> !Zp {
  %zero = mod_arith.constant 0 : !Zp
  %add = mod_arith.add %x, %zero : !Zp
  // CHECK: return %[[arg0]] : [[T]]
  return %add : !Zp
}

// CHECK-LABEL: @test_add_zero_lhs
// CHECK: (%[[arg0:.*]]: [[T:.*]]) -> [[T]]
func.func @test_add_zero_lhs(%x: !Zp) -> !Zp {
  %zero = mod_arith.constant 0 : !Zp
  %add = mod_arith.add %zero, %x : !Zp
  // CHECK: return %[[arg0]] : [[T]]
  return %add : !Zp
}

// CHECK-LABEL: @test_sub_zero
// CHECK: (%[[arg0:.*]]: [[T:.*]]) -> [[T]]
func.func @test_sub_zero(%x: !Zp) -> !Zp {
  %zero = mod_arith.constant 0 : !Zp
  %sub = mod_arith.sub %x, %zero : !Zp
  // CHECK: return %[[arg0]] : [[T]]
  return %sub : !Zp
}

// CHECK-LABEL: @test_mul_zero_rhs
// CHECK: (%[[arg0:.*]]: [[T:.*]]) -> [[T]]
func.func @test_mul_zero_rhs(%x: !Zp) -> !Zp {
  // CHECK: %[[res0:.+]] = mod_arith.constant 0 : [[T]]
  %zero = mod_arith.constant 0 : !Zp
  %mul = mod_arith.mul %x, %zero : !Zp
  // CHECK: return %[[res0]] : [[T]]
  return %mul : !Zp
}

// CHECK-LABEL: @test_mul_zero_lhs
// CHECK: (%[[arg0:.*]]: [[T:.*]]) -> [[T]]
func.func @test_mul_zero_lhs(%x: !Zp) -> !Zp {
  // CHECK: %[[res0:.+]] = mod_arith.constant 0 : [[T]]
  %zero = mod_arith.constant 0 : !Zp
  %mul = mod_arith.mul %zero, %x : !Zp
  // CHECK: return %[[res0]] : [[T]]
  return %mul : !Zp
}

// CHECK-LABEL: @test_mul_one_rhs
// CHECK: (%[[arg0:.*]]: [[T:.*]]) -> [[T]]
func.func @test_mul_one_rhs(%x: !Zp) -> !Zp {
  %one = mod_arith.constant 1 : !Zp
  %mul = mod_arith.mul %x, %one : !Zp
  // CHECK: return %[[arg0]] : [[T]]
  return %mul : !Zp
}

// CHECK-LABEL: @test_mul_one_lhs
// CHECK: (%[[arg0:.*]]: [[T:.*]]) -> [[T]]
func.func @test_mul_one_lhs(%x: !Zp) -> !Zp {
  %one = mod_arith.constant 1 : !Zp
  %mul = mod_arith.mul %one, %x : !Zp
  // CHECK: return %[[arg0]] : [[T]]
  return %mul : !Zp
}

// CHECK-LABEL: @test_add_add_const
// CHECK: (%[[arg0:.*]]: [[T:.*]]) -> [[T]]
func.func @test_add_add_const(%x: !Zp) -> !Zp {
  // CHECK: %[[res0:.+]] = mod_arith.constant 4 : [[T]]
  // CHECK: %[[res1:.+]] = mod_arith.add %[[arg0]], %[[res0]] : [[T]]
  %c0 = mod_arith.constant 12 : !Zp
  %c1 = mod_arith.constant 34 : !Zp
  %add = mod_arith.add %x, %c0 : !Zp
  %add2 = mod_arith.add %add, %c1 : !Zp
  // CHECK: return %[[res1]] : [[T]]
  return %add2 : !Zp
}

// CHECK-LABEL: @test_add_sub_const_rhs
// CHECK: (%[[arg0:.*]]: [[T:.*]]) -> [[T]]
func.func @test_add_sub_const_rhs(%x: !Zp) -> !Zp {
  // CHECK: %[[res0:.+]] = mod_arith.constant 22 : [[T]]
  // CHECK: %[[res1:.+]] = mod_arith.add %[[arg0]], %[[res0]] : [[T]]
  %c0 = mod_arith.constant 12 : !Zp
  %c1 = mod_arith.constant 34 : !Zp
  %sub = mod_arith.sub %x, %c0 : !Zp
  %add = mod_arith.add %sub, %c1 : !Zp
  // CHECK: return %[[res1]] : [[T]]
  return %add : !Zp
}

// CHECK-LABEL: @test_add_sub_const_lhs
// CHECK: (%[[arg0:.*]]: [[T:.*]]) -> [[T]]
func.func @test_add_sub_const_lhs(%x: !Zp) -> !Zp {
  // CHECK: %[[res0:.+]] = mod_arith.constant 4 : [[T]]
  // CHECK: %[[res1:.+]] = mod_arith.sub %[[res0]], %[[arg0]] : [[T]]
  %c0 = mod_arith.constant 12 : !Zp
  %c1 = mod_arith.constant 34 : !Zp
  %sub = mod_arith.sub %c0, %x : !Zp
  %add = mod_arith.add %sub, %c1 : !Zp
  // CHECK: return %[[res1]] : [[T]]
  return %add : !Zp
}

// CHECK-LABEL: @test_add_mul_neg_one_rhs
// CHECK: (%[[arg0:.*]]: [[T:.*]], %[[arg1:.*]]: [[T]]) -> [[T]]
func.func @test_add_mul_neg_one_rhs(%x: !Zp, %y: !Zp) -> !Zp {
  // CHECK: %[[res0:.+]] = mod_arith.sub %[[arg0]], %[[arg1]] : [[T]]
  %neg_one = mod_arith.constant 41 : !Zp
  %mul = mod_arith.mul %y, %neg_one : !Zp
  %add = mod_arith.add %x, %mul : !Zp
  // CHECK: return %[[res0]] : [[T]]
  return %add : !Zp
}

// CHECK-LABEL: @test_add_mul_neg_one_lhs
// CHECK: (%[[arg0:.*]]: [[T:.*]], %[[arg1:.*]]: [[T]]) -> [[T]]
func.func @test_add_mul_neg_one_lhs(%x: !Zp, %y: !Zp) -> !Zp {
  // CHECK: %[[res0:.+]] = mod_arith.sub %[[arg1]], %[[arg0]] : [[T]]
  %neg_one = mod_arith.constant 41 : !Zp
  %mul = mod_arith.mul %neg_one, %x : !Zp
  %add = mod_arith.add %mul, %y : !Zp
  // CHECK: return %[[res0]] : [[T]]
  return %add : !Zp
}

// CHECK-LABEL: @test_sub_mul_neg_one_rhs
// CHECK: (%[[arg0:.*]]: [[T:.*]], %[[arg1:.*]]: [[T]]) -> [[T]]
func.func @test_sub_mul_neg_one_rhs(%x: !Zp, %y: !Zp) -> !Zp {
  // CHECK: %[[res0:.+]] = mod_arith.add %[[arg0]], %[[arg1]] : [[T]]
  %neg_one = mod_arith.constant 41 : !Zp
  %mul = mod_arith.mul %y, %neg_one : !Zp
  %sub = mod_arith.sub %x, %mul : !Zp
  // CHECK: return %[[res0]] : [[T]]
  return %sub : !Zp
}

// CHECK-LABEL: @test_sub_mul_neg_one_lhs
// CHECK: (%[[arg0:.*]]: [[T:.*]], %[[arg1:.*]]: [[T]]) -> [[T]]
func.func @test_sub_mul_neg_one_lhs(%x: !Zp, %y: !Zp) -> !Zp {
  // CHECK: %[[res0:.+]] = mod_arith.constant 0 : [[T]]
  // CHECK: %[[res1:.+]] = mod_arith.add %[[arg0]], %[[arg1]] : [[T]]
  // CHECK: %[[res2:.+]] = mod_arith.sub %[[res0]], %[[res1]] : [[T]]
  %neg_one = mod_arith.constant 41 : !Zp
  %mul = mod_arith.mul %x, %neg_one : !Zp
  %sub = mod_arith.sub %mul, %y : !Zp
  // CHECK: return %[[res2]] : [[T]]
  return %sub : !Zp
}

// CHECK-LABEL: @test_mul_mul_const
// CHECK: (%[[arg0:.*]]: [[T:.*]]) -> [[T]]
func.func @test_mul_mul_const(%x: !Zp) -> !Zp {
  // CHECK: %[[res0:.+]] = mod_arith.constant 30 : [[T]]
  // CHECK: %[[res1:.+]] = mod_arith.mul %[[arg0]], %[[res0]] : [[T]]
  %c0 = mod_arith.constant 12 : !Zp
  %c1 = mod_arith.constant 34 : !Zp
  %mul = mod_arith.mul %x, %c0 : !Zp
  %mul2 = mod_arith.mul %mul, %c1 : !Zp
  // CHECK: return %[[res1]] : [[T]]
  return %mul2 : !Zp
}

// CHECK-LABEL: @test_sub_rhs_add_const
// CHECK: (%[[arg0:.*]]: [[T:.*]]) -> [[T]]
func.func @test_sub_rhs_add_const(%x: !Zp) -> !Zp {
  // CHECK: %[[res0:.+]] = mod_arith.constant 20 : [[T]]
  // CHECK: %[[res1:.+]] = mod_arith.add %[[arg0]], %[[res0]] : [[T]]
  %c0 = mod_arith.constant 12 : !Zp
  %c1 = mod_arith.constant 34 : !Zp
  %add = mod_arith.add %x, %c0 : !Zp
  %sub = mod_arith.sub %add, %c1 : !Zp
  // CHECK: return %[[res1]] : [[T]]
  return %sub : !Zp
}

// CHECK-LABEL: @test_sub_lhs_add_const
// CHECK: (%[[arg0:.*]]: [[T:.*]]) -> [[T]]
func.func @test_sub_lhs_add_const(%x: !Zp) -> !Zp {
  // CHECK: %[[res0:.+]] = mod_arith.constant 22 : [[T]]
  // CHECK: %[[res1:.+]] = mod_arith.sub %[[res0]], %[[arg0]] : [[T]]
  %c0 = mod_arith.constant 12 : !Zp
  %c1 = mod_arith.constant 34 : !Zp
  %add = mod_arith.add %x, %c0 : !Zp
  %sub = mod_arith.sub %c1, %add : !Zp
  // CHECK: return %[[res1]] : [[T]]
  return %sub : !Zp
}

// CHECK-LABEL: @test_sub_sub_lhs_rhs_lhs
// CHECK: (%[[arg0:.*]]: [[T:.*]], %[[arg1:.*]]: [[T]]) -> [[T]]
func.func @test_sub_sub_lhs_rhs_lhs(%a: !Zp, %b: !Zp) -> !Zp {
  // CHECK: %[[res0:.+]] = mod_arith.constant 0 : [[T]]
  // CHECK: %[[res1:.+]] = mod_arith.sub %[[res0]], %[[arg1]] : [[T]]
  %sub = mod_arith.sub %a, %b : !Zp
  %sub2 = mod_arith.sub %sub, %a : !Zp
  // CHECK: return %[[res1]] : [[T]]
  return %sub2 : !Zp
}
