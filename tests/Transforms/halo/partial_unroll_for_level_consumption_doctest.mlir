// RUN: heir-opt --partial-unroll-for-level-consumption=force-max-level=4 %s | FileCheck %s

// CHECK: @doctest
// CHECK: scf.for
// CHECK-NEXT: mgmt.bootstrap
// CHECK-NEXT: arith.muli
// CHECK-NEXT: mgmt.relinearize
// CHECK-NEXT: mgmt.modreduce
// CHECK-NEXT: arith.muli
// CHECK-NEXT: mgmt.relinearize
// CHECK-NEXT: mgmt.modreduce
// CHECK-NEXT: arith.muli
// CHECK-NEXT: mgmt.relinearize
// CHECK-NEXT: mgmt.modreduce
// CHECK-NEXT: arith.muli
// CHECK-NEXT: mgmt.relinearize
// CHECK-NEXT: mgmt.modreduce
// CHECK-NEXT: mgmt.level_reduce_min
// CHECK-NEXT: scf.yield
func.func @doctest(%arg0: !secret.secret<i32>) -> !secret.secret<i32> {
  %c1_i32 = arith.constant 1 : i32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c12 = arith.constant 12 : index
  %0 = secret.generic(%arg0: !secret.secret<i32>) {
  ^body(%input0: i32):
    %1 = arith.addi %c1_i32, %input0 : i32
    %2 = mgmt.level_reduce_min %1 : i32
    %3 = scf.for %arg1 = %c0 to %c12 step %c1 iter_args(%arg2 = %2) -> (i32) {
      %4 = mgmt.bootstrap %arg2 : i32
      %5 = arith.muli %4, %input0 : i32
      %6 = mgmt.relinearize %5 : i32
      %7 = mgmt.modreduce %6 : i32
      %8 = mgmt.level_reduce_min %7 : i32
      scf.yield %8 : i32
    }
    secret.yield %3 : i32
  } -> !secret.secret<i32>
  return %0 : !secret.secret<i32>
}
