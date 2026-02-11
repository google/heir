// RUN: heir-opt --secret-insert-mgmt-ckks=slot-number=1024 --tensor-linalg-to-affine-loops --arith-to-cggi --verify-diagnostics -split-input-file %s | FileCheck %s

// CHECK: func.func @scalar_op_combinations
// CHECK-SAME: %[[ARG0:.*]]: ![[CT:.*]], %[[ARG1:.*]]: ![[CT]]
func.func @scalar_op_combinations(%arg0: i32, %arg1: i32) -> i32 {
  // CHECK: %[[ADD:.*]] = cggi.add %[[ARG0]], %[[ARG1]]
  %0 = arith.addi %arg0, %arg1 : i32
  // CHECK: %[[MUL:.*]] = cggi.mul %[[ARG0]], %[[ARG1]]
  %1 = arith.muli %arg0, %arg1 : i32
  // CHECK: %[[CMP:.*]] = cggi.cmp %[[ADD]], %[[MUL]] {predicate = 2 : i64}
  %cond = arith.cmpi slt, %0, %1 : i32
  // CHECK: %[[SEL:.*]] = cggi.cmux %[[CMP]], %[[MUL]], %[[ARG0]]
  %2 = arith.select %cond, %1, %arg0 : i1, i32
  // CHECK: return %[[SEL]]
  return %2 : i32
}

// -----

func.func @scalar_mul(%arg0: !secret.secret<i16>) -> !secret.secret<i16> {
  %0 = secret.generic(%arg0 : !secret.secret<i16>) {
  ^bb0(%arg1: i16):
    %c2 = arith.constant 2 : i16
    // expected-error@+1 {{failed to legalize unresolved materialization}}
    %1 = arith.muli %arg1, %c2 : i16
    // expected-note@+1 {{see existing live user here}}
    secret.yield %1 : i16
  } -> (!secret.secret<i16>)
  return %0 : !secret.secret<i16>
}
