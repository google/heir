// RUN: heir-opt --validate-scale --verify-diagnostics --split-input-file %s

// Case 1: Valid bootstrap (should pass)
module attributes {ckks.schemeParam = #ckks.scheme_param<logN = 13, Q = [36028797019389953, 35184372121601], P = [36028797019488257], logDefaultScale = 45>, scheme.ckks} {
  func.func @bootstrap_success(%arg0: !secret.secret<f32> {mgmt.mgmt = #mgmt.mgmt<level = 1, scale = 54>}) -> !secret.secret<f32> {
    %0 = secret.generic(%arg0 : !secret.secret<f32> {mgmt.mgmt = #mgmt.mgmt<level = 1, scale = 54>}) {
    ^body(%input0: f32):
      // first-mod-bits = round(log2(36028797019389953)) = 55
      // input scale 54 <= 55 - 1 (54) -> OK
      // output scale 45 == logDefaultScale (45) -> OK
      %1 = mgmt.bootstrap %input0 {id = 0 : i64, mgmt.mgmt = #mgmt.mgmt<level = 1, scale = 45>} : f32
      secret.yield %1 : f32
    } -> (!secret.secret<f32> {mgmt.mgmt = #mgmt.mgmt<level = 1, scale = 45>})
    return %0 : !secret.secret<f32>
  }
}

// -----

// Case 2: Input scale too large
module attributes {ckks.schemeParam = #ckks.scheme_param<logN = 13, Q = [36028797019389953, 35184372121601], P = [36028797019488257], logDefaultScale = 45>, scheme.ckks} {
  func.func @bootstrap_input_scale_too_large(%arg0: !secret.secret<f32> {mgmt.mgmt = #mgmt.mgmt<level = 1, scale = 55>}) -> !secret.secret<f32> {
    %0 = secret.generic(%arg0 : !secret.secret<f32> {mgmt.mgmt = #mgmt.mgmt<level = 1, scale = 55>}) {
    ^body(%input0: f32):
      // first-mod-bits = 55
      // input scale 55 > 55 - 1 (54) -> Fail
      // expected-error @+1 {{input scale must be less than or equal to first-mod-bits - 1}}
      %1 = mgmt.bootstrap %input0 {id = 0 : i64, mgmt.mgmt = #mgmt.mgmt<level = 1, scale = 45>} : f32
      secret.yield %1 : f32
    } -> (!secret.secret<f32> {mgmt.mgmt = #mgmt.mgmt<level = 1, scale = 45>})
    return %0 : !secret.secret<f32>
  }
}

// -----

// Case 3: Output scale mismatch
module attributes {ckks.schemeParam = #ckks.scheme_param<logN = 13, Q = [36028797019389953, 35184372121601], P = [36028797019488257], logDefaultScale = 45>, scheme.ckks} {
  func.func @bootstrap_output_scale_mismatch(%arg0: !secret.secret<f32> {mgmt.mgmt = #mgmt.mgmt<level = 1, scale = 45>}) -> !secret.secret<f32> {
    %0 = secret.generic(%arg0 : !secret.secret<f32> {mgmt.mgmt = #mgmt.mgmt<level = 1, scale = 45>}) {
    ^body(%input0: f32):
      // output scale 46 != logDefaultScale (45) -> Fail
      // expected-error @+1 {{output scale must match the default scale}}
      %1 = mgmt.bootstrap %input0 {id = 0 : i64, mgmt.mgmt = #mgmt.mgmt<level = 1, scale = 46>} : f32
      secret.yield %1 : f32
    } -> (!secret.secret<f32> {mgmt.mgmt = #mgmt.mgmt<level = 1, scale = 46>})
    return %0 : !secret.secret<f32>
  }
}
