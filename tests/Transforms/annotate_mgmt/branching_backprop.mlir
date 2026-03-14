// RUN: heir-opt --annotate-mgmt %s

// This test ensures `meet` is implemented on the level lattice, or else
// backpropagation through region-branching ops will not work properly;
// the backprop through the if branch will annotate with level 2.

// CHECK: @test_scf_if_level_mismatch_init
// CHECK: [[C7:%.*]] = arith.constant 7
// CHECK: secret.generic
// CHECK:   scf.if
// CHECK:     mgmt.init [[C7]]
// CHECK-SAME:   level = 0
// CHECK-SAME:   dimension = 3
// CHECK:     scf.yield
// CHECK:   } else {
// CHECK:     scf.yield
// CHECK:   } {mgmt.mgmt = #mgmt.mgmt<level = 0, dimension = 3>}

func.func @test_scf_if_level_mismatch_init(%arg0: i1, %arg1: !secret.secret<i32>) -> !secret.secret<i32> {
  %cst = arith.constant 7 : i32
  %1 = secret.generic(%arg1 : !secret.secret<i32>) {
  ^body(%arg1_val: i32):
    %0 = scf.if %arg0 -> (i32) {
      %1 = mgmt.init %cst : i32
      scf.yield %1 : i32
    } else {
      %2 = mgmt.level_reduce %arg1_val {levelToDrop = 2} : i32
      %3 = arith.muli %2, %2 : i32
      scf.yield %3 : i32
    }
    secret.yield %0 : i32
  } -> !secret.secret<i32>
  return %1 : !secret.secret<i32>
}
