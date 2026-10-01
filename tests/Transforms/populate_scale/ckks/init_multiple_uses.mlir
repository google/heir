// RUN: heir-opt %s --populate-scale-ckks | FileCheck %s

module attributes {ckks.schemeParam = #ckks.scheme_param<logN = 13, Q = [36028797019389953, 35184372121601], P = [36028797019488257], logDefaultScale = 45>, scheme.ckks} {
  // CHECK: func.func @test_init_multiple_uses
  func.func @test_init_multiple_uses(%arg0: !secret.secret<f32> {mgmt.mgmt = #mgmt.mgmt<level = 1, scale = 45>}, %arg1: !secret.secret<f32> {mgmt.mgmt = #mgmt.mgmt<level = 1, scale = 45>}) -> (!secret.secret<f32> {mgmt.mgmt = #mgmt.mgmt<level = 1, scale = 90>}, !secret.secret<f32> {mgmt.mgmt = #mgmt.mgmt<level = 1, scale = 45>}) {
    %cst = arith.constant 7.000000e+00 : f32
    %0:2 = secret.generic(%arg0 : !secret.secret<f32>, %arg1 : !secret.secret<f32>) {
    ^body(%input0: f32, %input1: f32):
      // CHECK: %[[CST:.*]] = arith.constant 7.000000e+00 : f32
      // CHECK: %[[INIT_90:.*]] = mgmt.init %[[CST]] {mgmt.mgmt = #mgmt.mgmt<level = {{.*}}dimension = 3, scale = 90>} : f32
      // CHECK: %[[INIT_45:.*]] = mgmt.init %[[CST]] {mgmt.mgmt = #mgmt.mgmt<{{.*}}scale = 45>} : f32
      // CHECK: %[[MUL:.*]] = arith.mulf %[[INPUT0:.*]], %[[INPUT0]]
      // CHECK-SAME: scale = 90
      // CHECK: %[[ADD1:.*]] = arith.addf %[[MUL]], %[[INIT_90]]
      // CHECK: %[[ADD2:.*]] = arith.addf %[[INPUT1:.*]], %[[INIT_45]]
      // CHECK: secret.yield %[[ADD1]], %[[ADD2]]

      %init = mgmt.init %cst {mgmt.mgmt = #mgmt.mgmt<level = 1>} : f32
      %m = arith.mulf %input0, %input0 {mgmt.mgmt = #mgmt.mgmt<level = 1, dimension = 3>} : f32
      %add1 = arith.addf %m, %init : f32
      %add2 = arith.addf %input1, %init : f32
      secret.yield %add1, %add2 : f32, f32
    } -> (!secret.secret<f32>, !secret.secret<f32>)
    return %0#0, %0#1 : !secret.secret<f32>, !secret.secret<f32>
  }
}
