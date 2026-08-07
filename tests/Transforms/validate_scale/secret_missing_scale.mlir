// RUN: heir-opt --validate-scale --verify-diagnostics --split-input-file %s

module attributes {ckks.schemeParam = #ckks.scheme_param<logN = 13, Q = [36028797019389953, 35184372121601], P = [36028797019488257], logDefaultScale = 45>, scheme.ckks} {
  // expected-error @+1 {{secret block argument has no scale}}
  func.func @test_missing_arg_scale(%arg0: !secret.secret<f32>) -> !secret.secret<f32> {
    return %arg0 : !secret.secret<f32>
  }
}

// -----

module attributes {ckks.schemeParam = #ckks.scheme_param<logN = 13, Q = [36028797019389953, 35184372121601], P = [36028797019488257], logDefaultScale = 45>, scheme.ckks} {
  func.func @test_missing_result_scale(%arg0: !secret.secret<f32> {mgmt.mgmt = #mgmt.mgmt<level = 1, scale = 45>}) -> !secret.secret<f32> {
    // expected-error @+1 {{secret result value has no scale}}
    %0 = secret.generic(%arg0 : !secret.secret<f32>) {
    ^body(%input0: f32):
      %1 = arith.addf %input0, %input0 : f32
      secret.yield %1 : f32
    } -> (!secret.secret<f32>)
    return %0 : !secret.secret<f32>
  }
}
