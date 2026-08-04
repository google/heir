// RUN: not heir-opt --region-branch-op-level-invariance="level-budget=1" %s 2>&1 | FileCheck %s

// CHECK: value has invalid level
func.func @test_scf_if_level_mismatch(%arg0: i1, %arg1: !secret.secret<i32>) -> !secret.secret<i32> {
  %1 = secret.generic(%arg1 : !secret.secret<i32>) {
  ^body(%arg1_val: i32):
    %0 = scf.if %arg0 -> (i32) {
      %2 = mgmt.level_reduce %arg1_val {levelToDrop = 1} : i32
      scf.yield %2 : i32
    } else {
      %2 = mgmt.level_reduce %arg1_val {levelToDrop = 2} : i32
      scf.yield %2 : i32
    }
    secret.yield %0 : i32
  } -> !secret.secret<i32>
  return %1 : !secret.secret<i32>
}
