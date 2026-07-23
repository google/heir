// RUN: heir-opt --annotate-bootstrap-waterline="bootstrap-waterline=2" %s | FileCheck %s

module {
  // CHECK: func.func @main
  // CHECK-SAME: %[[arg0:.*]]: tensor<1x8xf32> {mgmt.bootstrap_waterline_level = 0 : i64, mgmt.needs_bootstrap = false, secret.secret}
  func.func @main(%arg0: tensor<1x8xf32> {secret.secret}) -> tensor<1x8xf32> {
    %cst = arith.constant dense<2.000000e+00> : tensor<1x8xf32>

    // CHECK: %[[m1:.*]] = arith.mulf
    // CHECK-SAME: {mgmt.bootstrap_waterline_level = 0 : i64, mgmt.needs_bootstrap = false}
    %m1 = arith.mulf %arg0, %cst : tensor<1x8xf32>

    // CHECK: %[[r1:.*]] = mgmt.modreduce
    // CHECK-SAME: {mgmt.bootstrap_waterline_level = 1 : i64, mgmt.needs_bootstrap = false}
    %r1 = mgmt.modreduce %m1 : tensor<1x8xf32>

    // CHECK: %[[m2:.*]] = arith.mulf
    // CHECK-SAME: {mgmt.bootstrap_waterline_level = 1 : i64, mgmt.needs_bootstrap = false}
    %m2 = arith.mulf %r1, %cst : tensor<1x8xf32>

    // CHECK: %[[r2:.*]] = mgmt.modreduce
    // CHECK-SAME: {mgmt.bootstrap_waterline_level = 2 : i64, mgmt.needs_bootstrap = false}
    %r2 = mgmt.modreduce %m2 : tensor<1x8xf32>

    // CHECK: %[[m3:.*]] = arith.mulf
    // CHECK-SAME: {mgmt.bootstrap_waterline_level = 2 : i64, mgmt.needs_bootstrap = false}
    %m3 = arith.mulf %r2, %cst : tensor<1x8xf32>

    // CHECK: %[[r3:.*]] = mgmt.modreduce
    // CHECK-SAME: {mgmt.bootstrap_waterline_level = 1 : i64, mgmt.needs_bootstrap = true}
    %r3 = mgmt.modreduce %m3 : tensor<1x8xf32>

    // CHECK: %[[m4:.*]] = arith.mulf
    // CHECK-SAME: {mgmt.bootstrap_waterline_level = 1 : i64, mgmt.needs_bootstrap = false}
    %m4 = arith.mulf %r3, %cst : tensor<1x8xf32>

    // CHECK: %[[r4:.*]] = mgmt.modreduce
    // CHECK-SAME: {mgmt.bootstrap_waterline_level = 2 : i64, mgmt.needs_bootstrap = false}
    %r4 = mgmt.modreduce %m4 : tensor<1x8xf32>

    return %r4 : tensor<1x8xf32>
  }

  // CHECK: func.func @test_bootstrap_reset
  // CHECK-SAME: %[[arg0:.*]]: tensor<1x8xf32> {mgmt.bootstrap_waterline_level = 0 : i64, mgmt.needs_bootstrap = false, secret.secret}
  func.func @test_bootstrap_reset(%arg0: tensor<1x8xf32> {secret.secret}) -> tensor<1x8xf32> {
    %cst = arith.constant dense<2.000000e+00> : tensor<1x8xf32>

    // CHECK: %[[m1:.*]] = arith.mulf
    // CHECK-SAME: {mgmt.bootstrap_waterline_level = 0 : i64, mgmt.needs_bootstrap = false}
    %m1 = arith.mulf %arg0, %cst : tensor<1x8xf32>

    // CHECK: %[[r1:.*]] = mgmt.modreduce
    // CHECK-SAME: {mgmt.bootstrap_waterline_level = 1 : i64, mgmt.needs_bootstrap = false}
    %r1 = mgmt.modreduce %m1 : tensor<1x8xf32>

    // CHECK: %[[m2:.*]] = arith.mulf
    // CHECK-SAME: {mgmt.bootstrap_waterline_level = 1 : i64, mgmt.needs_bootstrap = false}
    %m2 = arith.mulf %r1, %cst : tensor<1x8xf32>

    // CHECK: %[[r2:.*]] = mgmt.modreduce
    // CHECK-SAME: {mgmt.bootstrap_waterline_level = 2 : i64, mgmt.needs_bootstrap = false}
    %r2 = mgmt.modreduce %m2 : tensor<1x8xf32>

    // CHECK: %[[m3:.*]] = arith.mulf
    // CHECK-SAME: {mgmt.bootstrap_waterline_level = 2 : i64, mgmt.needs_bootstrap = false}
    %m3 = arith.mulf %r2, %cst : tensor<1x8xf32>

    // CHECK: %[[boot:.*]] = mgmt.bootstrap
    // CHECK-SAME: {mgmt.bootstrap_waterline_level = 0 : i64, mgmt.needs_bootstrap = false}
    %boot = mgmt.bootstrap %m3 : tensor<1x8xf32>

    // CHECK: %[[m4:.*]] = arith.mulf
    // CHECK-SAME: {mgmt.bootstrap_waterline_level = 0 : i64, mgmt.needs_bootstrap = false}
    %m4 = arith.mulf %boot, %cst : tensor<1x8xf32>

    // CHECK: %[[r3:.*]] = mgmt.modreduce
    // CHECK-SAME: {mgmt.bootstrap_waterline_level = 1 : i64, mgmt.needs_bootstrap = false}
    %r3 = mgmt.modreduce %m4 : tensor<1x8xf32>

    return %r3 : tensor<1x8xf32>
  }
}
