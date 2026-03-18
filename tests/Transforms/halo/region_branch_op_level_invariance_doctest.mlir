// RUN: heir-opt --region-branch-op-level-invariance %s | FileCheck %s

// CHECK: @doctest
func.func @doctest(%arg0: i1, %arg1: !secret.secret<i32>) -> !secret.secret<i32> {
  // CHECK: secret.generic
  // CHECK: scf.if
  // CHECK:   %[[MUL:.*]] = arith.muli
  // CHECK:   %[[RED:.*]] = mgmt.modreduce %[[MUL]]
  // CHECK:   scf.yield %[[RED]]
  // CHECK: else
  // CHECK:   %[[RED2:.*]] = mgmt.level_reduce
  // CHECK:   scf.yield %[[RED2]]
  %1 = secret.generic(%arg1 : !secret.secret<i32>) {
  ^body(%arg1_val: i32):
    %0 = scf.if %arg0 -> (i32) {
      %2 = arith.muli %arg1_val, %arg1_val : i32
      %3 = mgmt.modreduce %2 : i32
      scf.yield %3 : i32
    } else {
      scf.yield %arg1_val : i32
    }
    secret.yield %0 : i32
  } -> !secret.secret<i32>
  return %1 : !secret.secret<i32>
}
