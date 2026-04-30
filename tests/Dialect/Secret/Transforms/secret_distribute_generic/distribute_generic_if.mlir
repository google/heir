// RUN: heir-opt --wrap-generic --secret-distribute-generic="distribute-through=scf.if,affine.if" --split-input-file %s | FileCheck %s

// CHECK: @test_distribute_scf_if
// CHECK-SAME: %[[arg0:.*]]: !secret.secret<i32>, %[[cond:.*]]: i1
func.func @test_distribute_scf_if(%arg0: !secret.secret<i32>, %cond: i1) -> !secret.secret<i32> {
  // CHECK-NOT: secret.generic
  // CHECK: %[[res:.*]] = scf.if %[[cond]] -> (!secret.secret<i32>) {
  // CHECK:   %[[gen1:.*]] = secret.generic(%[[arg0]]: !secret.secret<i32>) {
  // CHECK:   ^body(%[[clear_arg0:.*]]: i32):
  // CHECK:     %[[add:.*]] = arith.addi %[[clear_arg0]], %[[clear_arg0]] : i32
  // CHECK:     secret.yield %[[add]] : i32
  // CHECK:   } -> !secret.secret<i32>
  // CHECK:   scf.yield %[[gen1]] : !secret.secret<i32>
  // CHECK: } else {
  // CHECK:   %[[gen2:.*]] = secret.generic(%[[arg0]]: !secret.secret<i32>) {
  // CHECK:   ^body(%[[clear_arg0:.*]]: i32):
  // CHECK:     %[[mul:.*]] = arith.muli %[[clear_arg0]], %[[clear_arg0]] : i32
  // CHECK:     secret.yield %[[mul]] : i32
  // CHECK:   } -> !secret.secret<i32>
  // CHECK:   scf.yield %[[gen2]] : !secret.secret<i32>
  // CHECK: }
  // CHECK: return %[[res]] : !secret.secret<i32>
  %0 = secret.generic (%arg0 : !secret.secret<i32>) {
  ^bb0(%clear_arg0: i32):
    %1 = scf.if %cond -> i32 {
      %2 = arith.addi %clear_arg0, %clear_arg0 : i32
      scf.yield %2 : i32
    } else {
      %3 = arith.muli %clear_arg0, %clear_arg0 : i32
      scf.yield %3 : i32
    }
    secret.yield %1 : i32
  } -> (!secret.secret<i32>)
  return %0 : !secret.secret<i32>
}

// -----

// CHECK: @test_distribute_affine_if
// CHECK-SAME: %[[arg0:.*]]: !secret.secret<i32>
func.func @test_distribute_affine_if(%arg0: !secret.secret<i32>) -> !secret.secret<i32> {
  %c0 = arith.constant 0 : index
  // CHECK-NOT: secret.generic
  // CHECK: %[[res:.*]] = affine.if
  // CHECK:   %[[gen1:.*]] = secret.generic(%[[arg0]]: !secret.secret<i32>) {
  // CHECK:   ^body(%[[clear_arg0:.*]]: i32):
  // CHECK:     %[[add:.*]] = arith.addi %[[clear_arg0]], %[[clear_arg0]] : i32
  // CHECK:     secret.yield %[[add]] : i32
  // CHECK:   } -> !secret.secret<i32>
  // CHECK:   affine.yield %[[gen1]] : !secret.secret<i32>
  // CHECK: } else {
  // CHECK:   %[[gen2:.*]] = secret.generic(%[[arg0]]: !secret.secret<i32>) {
  // CHECK:   ^body(%[[clear_arg0:.*]]: i32):
  // CHECK:     %[[mul:.*]] = arith.muli %[[clear_arg0]], %[[clear_arg0]] : i32
  // CHECK:     secret.yield %[[mul]] : i32
  // CHECK:   } -> !secret.secret<i32>
  // CHECK:   affine.yield %[[gen2]] : !secret.secret<i32>
  // CHECK: }
  // CHECK: return %[[res]] : !secret.secret<i32>
  %0 = secret.generic (%arg0 : !secret.secret<i32>) {
  ^bb0(%clear_arg0: i32):
    %1 = affine.if affine_set<(d0) : (d0 == 0)>(%c0) -> i32 {
      %2 = arith.addi %clear_arg0, %clear_arg0 : i32
      affine.yield %2 : i32
    } else {
      %3 = arith.muli %clear_arg0, %clear_arg0 : i32
      affine.yield %3 : i32
    }
    secret.yield %1 : i32
  } -> (!secret.secret<i32>)
  return %0 : !secret.secret<i32>
}

// -----

// Case similar to the user's failing example
// CHECK: @test_user_example
// CHECK-SAME: %[[arg0:.*]]: !secret.secret<tensor<1x1024xf32>>, %[[arg1:.*]]: !secret.secret<tensor<1x1024xf32>>, %[[arg2:.*]]: i1
func.func @test_user_example(%arg0: !secret.secret<tensor<1x1024xf32>>, %arg1: !secret.secret<tensor<1x1024xf32>>, %cond: i1) -> !secret.secret<tensor<1x1024xf32>> {
  // CHECK: %[[res:.*]] = scf.if %[[arg2]] -> (!secret.secret<tensor<1x1024xf32>>) {
  // CHECK:   %[[gen1:.*]] = secret.generic(%[[arg0]]: !secret.secret<tensor<1x1024xf32>>, %[[arg1]]: !secret.secret<tensor<1x1024xf32>>) {
  // CHECK:   ^body(%[[clear_arg0:.*]]: tensor<1x1024xf32>, %[[clear_arg1:.*]]: tensor<1x1024xf32>):
  // CHECK:     %[[add:.*]] = arith.addf %[[clear_arg0]], %[[clear_arg1]] : tensor<1x1024xf32>
  // CHECK:     secret.yield %[[add]] : tensor<1x1024xf32>
  // CHECK:   } -> !secret.secret<tensor<1x1024xf32>>
  // CHECK:   scf.yield %[[gen1]] : !secret.secret<tensor<1x1024xf32>>
  // CHECK: } else {
  // CHECK:   scf.yield %[[arg1]] : !secret.secret<tensor<1x1024xf32>>
  // CHECK: }
  %0 = secret.generic(%arg0 : !secret.secret<tensor<1x1024xf32>>, %arg1 : !secret.secret<tensor<1x1024xf32>>) {
  ^bb0(%clear_arg0: tensor<1x1024xf32>, %clear_arg1: tensor<1x1024xf32>):
    %1 = scf.if %cond -> tensor<1x1024xf32> {
      %2 = arith.addf %clear_arg0, %clear_arg1 : tensor<1x1024xf32>
      scf.yield %2 : tensor<1x1024xf32>
    } else {
      scf.yield %clear_arg1 : tensor<1x1024xf32>
    }
    secret.yield %1 : tensor<1x1024xf32>
  } -> (!secret.secret<tensor<1x1024xf32>>)
  return %0 : !secret.secret<tensor<1x1024xf32>>
}

// -----

// CHECK: @test_distribute_scf_if_with_attr
// CHECK-SAME: %[[arg0:.*]]: !secret.secret<i32>, %[[cond:.*]]: i1
func.func @test_distribute_scf_if_with_attr(%arg0: !secret.secret<i32>, %cond: i1) -> !secret.secret<i32> {
  // CHECK: scf.if %[[cond]] -> (!secret.secret<i32>) {
  // CHECK: } else {
  // CHECK: } {test.attr = "test"}
  %0 = secret.generic (%arg0 : !secret.secret<i32>) {
  ^bb0(%clear_arg0: i32):
    %1 = scf.if %cond -> i32 {
      %2 = arith.addi %clear_arg0, %clear_arg0 : i32
      scf.yield %2 : i32
    } else {
      %3 = arith.muli %clear_arg0, %clear_arg0 : i32
      scf.yield %3 : i32
    } {test.attr = "test"}
    secret.yield %1 : i32
  } -> (!secret.secret<i32>)
  return %0 : !secret.secret<i32>
}
