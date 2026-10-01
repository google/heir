// RUN: heir-opt --annotate-level="level-budget=16" %s | FileCheck %s

// CHECK: @test_uninit_propagation
func.func @test_uninit_propagation(%arg0: !secret.secret<i32>) -> !secret.secret<i32> {
  %c0 = arith.constant 0 : index
  %c10 = arith.constant 10 : index
  %c1 = arith.constant 1 : index
  %0 = secret.generic(%arg0 : !secret.secret<i32>) {
  ^body(%val: i32):
    %loop1 = scf.for %arg1 = %c0 to %c10 step %c1 iter_args(%arg2 = %val) -> (i32) {
      %b = mgmt.bootstrap %arg2 : i32
      %1 = mgmt.modreduce %b : i32
      scf.yield %1 : i32
    }
    // CHECK: scf.for
    %loop2 = scf.for %arg3 = %c0 to %c10 step %c1 iter_args(%arg4 = %loop1) -> (i32) {
      // CHECK: mgmt.level_reduce_min
      // CHECK-SAME: {mgmt.level = "max"}
      %2 = mgmt.level_reduce_min %arg4 : i32
      scf.yield %2 : i32
    }
    secret.yield %loop2 : i32
  } -> !secret.secret<i32>
  return %0 : !secret.secret<i32>
}
