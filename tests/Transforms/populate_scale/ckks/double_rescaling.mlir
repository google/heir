// RUN: heir-opt %s --populate-scale-ckks | FileCheck %s

// Level 0 has two 30-bit primes (~60 bits), Level 1 has two 28-bit primes (268468225 * 268369921 ~ 2^56).
module attributes {ckks.schemeParam = #ckks.scheme_param<logN = 13, Q = [1073692673, 1073643521, 268468225, 268369921], P = [1073709057, 1073758209], logDefaultScale = 56, moduliPerLevel = [2, 2]>, scheme.ckks} {
  // CHECK: func.func @mul_double_rescale
  func.func @mul_double_rescale(%arg0: !secret.secret<f32>) -> !secret.secret<f32> {
    %0 = secret.generic(%arg0 : !secret.secret<f32> {mgmt.mgmt = #mgmt.mgmt<level = 1, scale = 56>}) {
    ^body(%input0: f32):
      %1 = arith.mulf %input0, %input0 {mgmt.mgmt = #mgmt.mgmt<level = 1, dimension = 3, scale = 112>} : f32
      // CHECK: mgmt.relinearize
      // CHECK-SAME: scale = 112
      %2 = mgmt.relinearize %1 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : f32
      // CHECK: mgmt.modreduce
      // CHECK-SAME: scale = 56
      %3 = mgmt.modreduce %2 {mgmt.mgmt = #mgmt.mgmt<level = 0>} : f32
      secret.yield %3 : f32
    } -> (!secret.secret<f32> {mgmt.mgmt = #mgmt.mgmt<level = 0>})
    return %0 : !secret.secret<f32>
  }
}
