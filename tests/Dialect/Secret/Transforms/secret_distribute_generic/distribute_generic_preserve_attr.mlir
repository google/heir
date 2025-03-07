// RUN: heir-opt --secret-distribute-generic %s | FileCheck %s

// CHECK-LABEL: test_distribute_generic_preserve_attr
// CHECK-SAME: %[[value:.*]]: !secret.secret<i32> {dialect.attr = 2 : i64}, %[[cond:.*]]: i1) -> !secret.secret<i32> {
func.func @test_distribute_generic_preserve_attr(%value: !secret.secret<i32>, %cond: i1) -> !secret.secret<i32> {
  // CHECK-NEXT: %[[g0:.*]] = secret.generic ins(%[[value]] : !secret.secret<i32>) {
  // CHECK-NEXT: ^[[bb0:.*]](%[[clear_g0_in0:.*]]: i32):
  // CHECK-NEXT:   %[[g0_op:.*]] = arith.muli %[[clear_g0_in0]], %[[clear_g0_in0]] {dialect.attr = 1 : i64} : i32
  // CHECK-NEXT:   secret.yield %[[g0_op]] : i32
  // CHECK-NEXT: } -> !secret.secret<i32>

  // CHECK-NEXT: return %[[g0]] : !secret.secret<i32>
  %Z = secret.generic
    ins(%value : !secret.secret<i32>) attrs = {__argattrs = [{dialect.attr = 2}]} {
    ^bb0(%clear_value: i32):
      %0 = arith.muli %clear_value, %clear_value {dialect.attr = 1} : i32
      secret.yield %0 : i32
    } -> (!secret.secret<i32>)
  func.return %Z : !secret.secret<i32>
}
