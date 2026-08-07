// RUN: heir-opt --validate-scale --verify-diagnostics --split-input-file %s

module attributes {ckks.schemeParam = #ckks.scheme_param<logN = 13, Q = [36028797019389953, 35184372121601], P = [36028797019488257], logDefaultScale = 45>, scheme.ckks} {
  func.func @adjust_scale_fail(%arg0: !secret.secret<f32> {mgmt.mgmt = #mgmt.mgmt<level = 1, scale = 90>}) -> !secret.secret<f32> {
    %0 = secret.generic(%arg0 : !secret.secret<f32> {mgmt.mgmt = #mgmt.mgmt<level = 1, scale = 90>}) {
    ^body(%input0: f32):
      // expected-error @+1 {{target scale must be greater than or equal to input scale}}
      %1 = mgmt.adjust_scale %input0 {id = 0 : i64, mgmt.mgmt = #mgmt.mgmt<level = 1, scale = 45>} : f32
      secret.yield %1 : f32
    } -> (!secret.secret<f32> {mgmt.mgmt = #mgmt.mgmt<level = 1, scale = 45>})
    return %0 : !secret.secret<f32>
  }
}
