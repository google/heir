// RUN: heir-opt --region-branch-op-level-invariance %s | FileCheck %s

// CHECK: @test_scf_if_level_mismatch
func.func @test_scf_if_level_mismatch(%arg0: i1, %arg1: !secret.secret<i32>) -> !secret.secret<i32> {
  // CHECK: secret.generic
  // CHECK: scf.if
  // CHECK:   %[[L1:.*]] = mgmt.level_reduce
  // CHECK:   %[[L2:.*]] = mgmt.level_reduce %[[L1]]
  // CHECK:   scf.yield %[[L2]]
  // CHECK: else
  // CHECK:   %[[L3:.*]] = mgmt.level_reduce %{{.*}} {levelToDrop = 2
  // CHECK:   scf.yield %[[L3]]
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

// CHECK: @test_affine_if_level_mismatch
func.func @test_affine_if_level_mismatch(%arg0: index, %arg1: !secret.secret<i32>) -> !secret.secret<i32> {
  // CHECK: secret.generic
  // CHECK: affine.if
  // CHECK:   %[[L1:.*]] = mgmt.level_reduce
  // CHECK:   %[[L2:.*]] = mgmt.level_reduce %[[L1]]
  // CHECK:   affine.yield %[[L2]]
  // CHECK: else
  // CHECK:   %[[L3:.*]] = mgmt.level_reduce %{{.*}} {levelToDrop = 2
  // CHECK:   affine.yield %[[L3]]
  %1 = secret.generic(%arg1 : !secret.secret<i32>) {
  ^body(%arg1_val: i32):
    %0 = affine.if affine_set<(d0) : (d0 == 0)>(%arg0) -> (i32) {
      %2 = mgmt.level_reduce %arg1_val {levelToDrop = 1} : i32
      affine.yield %2 : i32
    } else {
      %2 = mgmt.level_reduce %arg1_val {levelToDrop = 2} : i32
      affine.yield %2 : i32
    }
    secret.yield %0 : i32
  } -> !secret.secret<i32>
  return %1 : !secret.secret<i32>
}

// CHECK: @test_scf_if_level_mismatch_else_insertion
func.func @test_scf_if_level_mismatch_else_insertion(%arg0: i1, %arg1: !secret.secret<i32>) -> !secret.secret<i32> {
  // CHECK: secret.generic
  // CHECK: scf.if
  // CHECK:   %[[L1:.*]] = mgmt.level_reduce %{{.*}} {levelToDrop = 2
  // CHECK:   scf.yield %[[L1]]
  // CHECK: else
  // CHECK:   %[[L2:.*]] = mgmt.level_reduce
  // CHECK:   %[[L3:.*]] = mgmt.level_reduce %[[L2]]
  // CHECK:   scf.yield %[[L3]]
  %1 = secret.generic(%arg1 : !secret.secret<i32>) {
  ^body(%arg1_val: i32):
    %0 = scf.if %arg0 -> (i32) {
      %2 = mgmt.level_reduce %arg1_val {levelToDrop = 2} : i32
      scf.yield %2 : i32
    } else {
      %2 = mgmt.level_reduce %arg1_val {levelToDrop = 1} : i32
      scf.yield %2 : i32
    }
    secret.yield %0 : i32
  } -> !secret.secret<i32>
  return %1 : !secret.secret<i32>
}

// CHECK: @test_scf_if_realistic_mismatch
func.func @test_scf_if_realistic_mismatch(%arg0: i1, %arg1: !secret.secret<i32>) -> !secret.secret<i32> {
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
