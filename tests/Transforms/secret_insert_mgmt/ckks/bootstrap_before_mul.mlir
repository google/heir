// RUN: heir-opt --secret-insert-mgmt-ckks="bootstrap-waterline=2 level-budget=2 slot-number=8 after-mul=true" %s | FileCheck %s

module {
  // CHECK: func.func @main
  func.func @main(%arg0: tensor<1x8xf32> {secret.secret}) -> tensor<1x8xf32> {
    %cst = arith.constant dense<2.000000e+00> : tensor<1x8xf32>

    // Fresh input %arg0 is at level 2 (budget 2).
    // Mul 1: 2 * 2 -> 2. Modreduce -> 1.
    // CHECK: %[[m1:.*]] = arith.mulf
    // CHECK: mgmt.level_reduce %[[m1]]
    // CHECK: mgmt.modreduce
    %m1 = arith.mulf %arg0, %cst : tensor<1x8xf32>
    %r1 = mgmt.modreduce %m1 : tensor<1x8xf32>

    // CHECK: %[[m2:.*]] = arith.mulf
    // CHECK: %[[boot2:.*]] = mgmt.bootstrap %[[m2]]
    // CHECK: %[[lr2:.*]] = mgmt.level_reduce %[[boot2]]
    // CHECK: %[[adj2:.*]] = mgmt.adjust_scale %[[lr2]] {id = 0 : i64, mgmt.mgmt = #mgmt.mgmt<level = 1>}
    // CHECK: mgmt.modreduce %[[adj2]]
    %m2 = arith.mulf %r1, %cst : tensor<1x8xf32>
    %r2 = mgmt.modreduce %m2 : tensor<1x8xf32>

    // CHECK: %[[m3:.*]] = arith.mulf
    // CHECK: %[[boot3:.*]] = mgmt.bootstrap %[[m3]]
    // CHECK: %[[lr3:.*]] = mgmt.level_reduce %[[boot3]]
    // CHECK: %[[adj3:.*]] = mgmt.adjust_scale %[[lr3]] {id = 1 : i64, mgmt.mgmt = #mgmt.mgmt<level = 1>}
    // CHECK: mgmt.modreduce %[[adj3]]
    %m3 = arith.mulf %r2, %cst : tensor<1x8xf32>
    %r3 = mgmt.modreduce %m3 : tensor<1x8xf32>

    return %r3 : tensor<1x8xf32>
  }
}
