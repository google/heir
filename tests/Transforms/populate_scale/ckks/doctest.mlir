// RUN: heir-opt %s --populate-scale-ckks | FileCheck %s

module attributes {ckks.schemeParam = #ckks.scheme_param<logN = 13, Q = [36028797019389953, 35184372121601], P = [36028797019488257], logDefaultScale = 45>, scheme.ckks} {
  func.func @mul(%arg0: !secret.secret<f32>) -> !secret.secret<f32> {
    %0 = secret.generic(%arg0 : !secret.secret<f32> {mgmt.mgmt = #mgmt.mgmt<level = 1, scale = 45>}) {
    ^body(%input0: f32):
      %1 = arith.mulf %input0, %input0 {mgmt.mgmt = #mgmt.mgmt<level = 1, dimension = 3, scale = 90>} : f32
      // CHECK: mgmt.relinearize
      // CHECK-SAME: scale = 90
      %2 = mgmt.relinearize %1 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : f32
      // CHECK: mgmt.modreduce
      // CHECK-SAME: scale = 45
      %3 = mgmt.modreduce %2 {mgmt.mgmt = #mgmt.mgmt<level = 0>} : f32
      secret.yield %3 : f32
    } -> (!secret.secret<f32> {mgmt.mgmt = #mgmt.mgmt<level = 0>})
    return %0 : !secret.secret<f32>
  }
}
