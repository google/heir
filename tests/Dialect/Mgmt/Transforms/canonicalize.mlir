// RUN: heir-opt --canonicalize %s | FileCheck %s

// CHECK: func.func @test_swap
func.func @test_swap(%arg0: tensor<1x4096xf32>) -> tensor<1x4096xf32> {
  // CHECK: %[[INIT:.*]] = mgmt.init
  // CHECK: %[[LR:.*]] = mgmt.level_reduce %[[INIT]] {levelToDrop = 2 : i64, mgmt.mgmt = #mgmt.mgmt<level = 1, scale = 90>}
  // CHECK: %[[MR:.*]] = mgmt.modreduce %[[LR]] {mgmt.mgmt = #mgmt.mgmt<level = 0, scale = 45>}
  // CHECK: return %[[MR]]

  %input = mgmt.init %arg0 {mgmt.mgmt = #mgmt.mgmt<level = 3, scale = 90>} : tensor<1x4096xf32>
  %0 = mgmt.modreduce %input {mgmt.mgmt = #mgmt.mgmt<level = 2, scale = 45>} : tensor<1x4096xf32>
  %1 = mgmt.level_reduce %0 {levelToDrop = 2 : i64, mgmt.mgmt = #mgmt.mgmt<level = 0, scale = 45>} : tensor<1x4096xf32>
  return %1 : tensor<1x4096xf32>
}

// CHECK: func.func @test_swap_as_mr
func.func @test_swap_as_mr(%arg0: tensor<1x4096xf32>) -> tensor<1x4096xf32> {
  // CHECK: %[[INIT:.*]] = mgmt.init
  // CHECK: %[[AS:.*]] = mgmt.adjust_scale %[[INIT]] {id = 1 : i64, mgmt.mgmt = #mgmt.mgmt<level = 3, scale = 120>}
  // CHECK: %[[MR:.*]] = mgmt.modreduce %[[AS]] {mgmt.mgmt = #mgmt.mgmt<level = 2, scale = 60>}
  // CHECK: return %[[MR]]

  %input = mgmt.init %arg0 {mgmt.mgmt = #mgmt.mgmt<level = 3, scale = 90>} : tensor<1x4096xf32>
  %0 = mgmt.modreduce %input {mgmt.mgmt = #mgmt.mgmt<level = 2, scale = 45>} : tensor<1x4096xf32>
  %1 = mgmt.adjust_scale %0 {id = 1 : i64, mgmt.mgmt = #mgmt.mgmt<level = 2, scale = 60>} : tensor<1x4096xf32>
  return %1 : tensor<1x4096xf32>
}

// CHECK: func.func @test_swap_lr_as
func.func @test_swap_lr_as(%arg0: tensor<1x4096xf32>) -> tensor<1x4096xf32> {
  // CHECK: %[[INIT:.*]] = mgmt.init
  // CHECK: %[[LR:.*]] = mgmt.level_reduce %[[INIT]] {levelToDrop = 2 : i64, mgmt.mgmt = #mgmt.mgmt<level = 1, scale = 90>}
  // CHECK: %[[AS:.*]] = mgmt.adjust_scale %[[LR]] {id = 1 : i64, mgmt.mgmt = #mgmt.mgmt<level = 1, scale = 60>}
  // CHECK: return %[[AS]]

  %input = mgmt.init %arg0 {mgmt.mgmt = #mgmt.mgmt<level = 3, scale = 90>} : tensor<1x4096xf32>
  %0 = mgmt.adjust_scale %input {id = 1 : i64, mgmt.mgmt = #mgmt.mgmt<level = 3, scale = 60>} : tensor<1x4096xf32>
  %1 = mgmt.level_reduce %0 {levelToDrop = 2 : i64, mgmt.mgmt = #mgmt.mgmt<level = 1, scale = 60>} : tensor<1x4096xf32>
  return %1 : tensor<1x4096xf32>
}

// CHECK: func.func @test_merge_lr
func.func @test_merge_lr(%arg0: tensor<1x4096xf32>) -> tensor<1x4096xf32> {
  // CHECK: %[[INIT:.*]] = mgmt.init
  // CHECK: %[[LR:.*]] = mgmt.level_reduce %[[INIT]] {levelToDrop = 3 : i64, mgmt.mgmt = #mgmt.mgmt<level = 0, scale = 90>}
  // CHECK: return %[[LR]]

  %input = mgmt.init %arg0 {mgmt.mgmt = #mgmt.mgmt<level = 3, scale = 90>} : tensor<1x4096xf32>
  %0 = mgmt.level_reduce %input {levelToDrop = 1 : i64, mgmt.mgmt = #mgmt.mgmt<level = 2, scale = 90>} : tensor<1x4096xf32>
  %1 = mgmt.level_reduce %0 {levelToDrop = 2 : i64, mgmt.mgmt = #mgmt.mgmt<level = 0, scale = 90>} : tensor<1x4096xf32>
  return %1 : tensor<1x4096xf32>
}

// CHECK: func.func @test_merge_mr
func.func @test_merge_mr(%arg0: tensor<1x4096xf32>) -> tensor<1x4096xf32> {
  // CHECK: %[[INIT:.*]] = mgmt.init
  // CHECK: %[[LR:.*]] = mgmt.level_reduce %[[INIT]] {mgmt.mgmt = #mgmt.mgmt<level = 2, scale = 90>}
  // CHECK: %[[MR:.*]] = mgmt.modreduce %[[LR]] {mgmt.mgmt = #mgmt.mgmt<level = 1, scale = 45>}
  // CHECK: return %[[MR]]

  %input = mgmt.init %arg0 {mgmt.mgmt = #mgmt.mgmt<level = 3, scale = 90>} : tensor<1x4096xf32>
  %0 = mgmt.modreduce %input {mgmt.mgmt = #mgmt.mgmt<level = 2, scale = 45>} : tensor<1x4096xf32>
  %1 = mgmt.modreduce %0 {mgmt.mgmt = #mgmt.mgmt<level = 1, scale = 45>} : tensor<1x4096xf32>
  return %1 : tensor<1x4096xf32>
}

// CHECK: func.func @test_merge_as
func.func @test_merge_as(%arg0: tensor<1x4096xf32>) -> tensor<1x4096xf32> {
  // CHECK: %[[INIT:.*]] = mgmt.init
  // CHECK: %[[AS:.*]] = mgmt.adjust_scale %[[INIT]] {id = 2 : i64, mgmt.mgmt = #mgmt.mgmt<level = 3, scale = 80>}
  // CHECK: return %[[AS]]

  %input = mgmt.init %arg0 {mgmt.mgmt = #mgmt.mgmt<level = 3, scale = 90>} : tensor<1x4096xf32>
  %0 = mgmt.adjust_scale %input {id = 1 : i64, mgmt.mgmt = #mgmt.mgmt<level = 3, scale = 60>} : tensor<1x4096xf32>
  %1 = mgmt.adjust_scale %0 {id = 2 : i64, mgmt.mgmt = #mgmt.mgmt<level = 3, scale = 80>} : tensor<1x4096xf32>
  return %1 : tensor<1x4096xf32>
}
