// RUN: heir-opt --partial-unroll-for-level-consumption %s | FileCheck %s

// CHECK: @doctest([[arg0:%[^:]*]]: !secret.secret<i32>)
// CHECK:     affine.for [[i:[^ ]*]] = 1 to 3 iter_args([[iter_arg:%[^ ]*]] = [[level_reduced]]) -> (i32) {
// CHECK:       mgmt.bootstrap [[iter_arg]]
// CHECK-NEXT:  arith.addi
// CHECK-NEXT:  arith.addi
// CHECK-NEXT:  arith.addi
// CHECK-NEXT:  arith.addi
// CHECK:       mgmt.level_reduce_min
// CHECK:       affine.yield
func.func @doctest(%arg0: !secret.secret<i32>) -> !secret.secret<i32> {
  %c1_i32 = arith.constant 1 : i32
  %0 = secret.generic(%arg0: !secret.secret<i32>) {
  ^body(%input0: i32):
    %1 = arith.addi %c1_i32, %input0 : i32
    %2 = mgmt.level_reduce_min %1 : i32
    %3 = affine.for %arg1 = 1 to 12 iter_args(%arg2 = %2) -> (i32) {
      %4 = mgmt.bootstrap %arg2 : i32
      %5 = arith.muli %4, %input0 : i32
      %6 = mgmt.level_reduce_min %5 : i32
      affine.yield %6 : i32
    }
    secret.yield %3 : i32
  } -> !secret.secret<i32>
  return %0 : !secret.secret<i32>
}
