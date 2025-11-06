// RUN: heir-opt %s --populate-scale-ckks | FileCheck %s

// High-precision scale management: uses actual scales (2^logDefaultScale) instead of log scales
// logDefaultScale = 45, so default scale = 2^45 = 35184372088832
// After multiply: scale = 2^45 * 2^45 = 2^90 = 1237940039285380274899124224
// After modreduce (rescale by Q[1]): scale = 2^90 / 35184372121601 ≈ 35184372088832
module attributes {ckks.schemeParam = #ckks.scheme_param<logN = 13, Q = [36028797019389953, 35184372121601], P = [36028797019488257], logDefaultScale = 45>, scheme.ckks} {
  func.func @mul(%arg0: !secret.secret<f32>) -> !secret.secret<f32> {
    %0 = secret.generic(%arg0 : !secret.secret<f32> {mgmt.mgmt = #mgmt.mgmt<level = 1, scale = 0>}) {
    ^body(%input0: f32):
      %1 = arith.mulf %input0, %input0 {mgmt.mgmt = #mgmt.mgmt<level = 1, dimension = 3, scale = 0>} : f32
      // CHECK: mgmt.relinearize
      // CHECK-SAME: scale = 1237940039285380274899124224
      %2 = mgmt.relinearize %1 {mgmt.mgmt = #mgmt.mgmt<level = 1, scale = 0>} : f32
      // CHECK: mgmt.modreduce
      // CHECK-SAME: scale = 35184372088832
      %3 = mgmt.modreduce %2 {mgmt.mgmt = #mgmt.mgmt<level = 0, scale = 0>} : f32
      secret.yield %3 : f32
    } -> (!secret.secret<f32> {mgmt.mgmt = #mgmt.mgmt<level = 0, scale = 0>})
    return %0 : !secret.secret<f32>
  }
}
