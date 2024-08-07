// RUN: heir-opt --convert-secret-while-to-static-for %s | FileCheck %s

// CHECK-LABEL: @basic_while_loop_with_secret_condition
func.func @basic_while_loop_with_secret_condition(%input: !secret.secret<i16>) -> !secret.secret<i16> {
  // CHECK-NOT: scf.while
  // CHECK: %[[RESULT:.*]] = secret.generic ins(%[[SECRET_INPUT:.*]]: !secret.secret<i16>)
  // CHECK-NEXT: ^bb0(%[[INPUT:.*]]: i16):
  // CHECK: %[[FOR:.*]] = affine.for %[[I:.*]] = 0 to 16 iter_args(%[[ARG:.*]] = %[[INPUT]]) -> (i16)
  // CHECK-NEXT: arith.cmpi
  %c100 = arith.constant 100 : i16
  %c20 = arith.constant 20 : i16
  %0 = secret.generic ins(%input : !secret.secret<i16>) {
  ^bb0(%arg1: i16):
    %1 = scf.while (%arg2 = %arg1) : (i16) -> i16 {
      %3 = arith.cmpi sgt, %arg2, %c100 : i16
      scf.condition(%3) %arg2 : i16
    } do {
    ^bb0(%arg2: i16):
      %2 = arith.muli %arg2, %arg2 : i16
      scf.yield %2 : i16
    } attributes {max_iter = 16 : i64}
    secret.yield %1 : i16
  } -> !secret.secret<i16>
  return %0 : !secret.secret<i16>
}

// CHECK-LABEL: @while_loop_with_joint_secret_condition
func.func @while_loop_with_joint_secret_condition(%input: !secret.secret<i16>) -> !secret.secret<i16> {
  // CHECK-NOT: scf.while
  // CHECK: %[[RESULT:.*]] = secret.generic ins(%[[SECRET_INPUT:.*]]: !secret.secret<i16>)
  // CHECK-NEXT: ^bb0(%[[INPUT:.*]]: i16):
  // CHECK: %[[FOR:.*]] = affine.for %[[I:.*]] = 0 to 16 iter_args(%[[ARG:.*]] = %[[INPUT]]) -> (i16)
  // CHECK: arith.andi
  %c100 = arith.constant 100 : i16
  %c20 = arith.constant 20 : i16
  %0 = secret.generic ins(%input : !secret.secret<i16>) {
  ^bb0(%arg1: i16):
    %1 = scf.while (%arg2 = %arg1) : (i16) -> i16 {
      %3 = arith.cmpi slt, %arg2, %c100 : i16
      %4 = arith.cmpi sgt, %arg2, %c20 : i16
      %5 = arith.andi %3, %4 : i1
      scf.condition(%5) %arg2 : i16
    } do {
    ^bb0(%arg2: i16):
      %2 = arith.muli %arg2, %arg2 : i16
      scf.yield %2 : i16
    } attributes {max_iter = 16 : i64}
    secret.yield %1 : i16
  } -> !secret.secret<i16>
  return %0 : !secret.secret<i16>
}
