// RUN: heir-opt --validate-scale --verify-diagnostics --split-input-file %s

module attributes {ckks.schemeParam = #ckks.scheme_param<logN = 13, Q = [36028797019389953, 35184372121601], P = [36028797019488257], logDefaultScale = 45>, scheme.ckks} {
  func.func @mul_mismatch(%arg0: !secret.secret<f32> {mgmt.mgmt = #mgmt.mgmt<level = 1, scale = 45>}, %arg1: !secret.secret<f32> {mgmt.mgmt = #mgmt.mgmt<level = 1, scale = 45>}) -> !secret.secret<f32> {
    %0 = secret.generic(%arg0 : !secret.secret<f32> {mgmt.mgmt = #mgmt.mgmt<level = 1, scale = 45>}, %arg1 : !secret.secret<f32> {mgmt.mgmt = #mgmt.mgmt<level = 1, scale = 45>}) {
    ^body(%input0: f32, %input1: f32):
      // expected-error @+1 {{result scale must equal the sum of operand scales}}
      %1 = arith.mulf %input0, %input1 {mgmt.mgmt = #mgmt.mgmt<level = 1, scale = 80>} : f32
      secret.yield %1 : f32
    } -> (!secret.secret<f32> {mgmt.mgmt = #mgmt.mgmt<level = 1, scale = 80>})
    return %0 : !secret.secret<f32>
  }
}
