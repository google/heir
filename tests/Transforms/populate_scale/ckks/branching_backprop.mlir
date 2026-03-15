// RUN: heir-opt %s --populate-scale-ckks | FileCheck %s

// This test ensures `meet` is implemented on scale lattice, or else
// backpropagation through region-branching ops will not work properly.

module attributes {ckks.schemeParam = #ckks.scheme_param<logN = 13, Q = [36028797019389953, 35184372121601], P = [36028797019488257], logDefaultScale = 45>, scheme.ckks} {
  func.func @test_scf_if_scale_mismatch_init(%arg0: i1, %arg1: !secret.secret<f32> {mgmt.mgmt = #mgmt.mgmt<level = 1, scale = 45>}) -> !secret.secret<f32> {
    %cst = arith.constant 7.0 : f32
    %1 = secret.generic(%arg1 : !secret.secret<f32>) {
    ^body(%arg1_val: f32):
      %0 = scf.if %arg0 -> (f32) {
        // CHECK: mgmt.init
        // CHECK-SAME: scale = 90
        %1 = mgmt.init %cst {mgmt.mgmt = #mgmt.mgmt<level = 1>} : f32
        scf.yield %1 : f32
      } else {
        // CHECK: arith.mulf
        // CHECK-SAME: scale = 90
        %1 = arith.mulf %arg1_val, %arg1_val {mgmt.mgmt = #mgmt.mgmt<level = 1, dimension = 3>} : f32
        scf.yield %1 : f32
      }
      secret.yield %0 : f32
    } -> (!secret.secret<f32> {mgmt.mgmt = #mgmt.mgmt<level = 1, dimension = 3>})
    return %1 : !secret.secret<f32>
  }
}
