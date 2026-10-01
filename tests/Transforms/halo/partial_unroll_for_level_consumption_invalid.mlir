// RUN: not heir-opt --partial-unroll-for-level-consumption=force-max-level=1 %s 2>&1 | FileCheck %s

// CHECK: value has invalid level
func.func @affine_loop(%arg0: !secret.secret<i32>) -> !secret.secret<i32> {
  %c1_i32 = arith.constant 1 : i32
  %0 = secret.generic(%arg0: !secret.secret<i32>) {
  ^body(%input0: i32):
    %1 = arith.addi %c1_i32, %input0 : i32
    %2 = mgmt.level_reduce_min %1 : i32
    %3 = affine.for %arg1 = 1 to 12 iter_args(%arg2 = %2) -> (i32) {
      %4 = mgmt.bootstrap %arg2 : i32
      %5 = arith.muli %4, %input0 : i32
      %6 = mgmt.relinearize %5 : i32
      %7 = mgmt.modreduce %6 : i32
      %8 = arith.muli %7, %input0 : i32
      %9 = mgmt.relinearize %8 : i32
      %10 = mgmt.modreduce %9 : i32
      %11 = mgmt.level_reduce_min %10 : i32
      affine.yield %11 : i32
    }
    secret.yield %3 : i32
  } -> !secret.secret<i32>
  return %0 : !secret.secret<i32>
}
