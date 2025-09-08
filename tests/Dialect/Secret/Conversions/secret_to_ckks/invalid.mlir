// RUN: heir-opt --split-input-file --secret-to-ckks="poly-mod-degree=1024" --verify-diagnostics %s | FileCheck %s

// Tests invalid secret types

#mgmt = #mgmt.mgmt<level = 0, dimension = 2>
module attributes {ckks.schemeParam = #ckks.scheme_param<logN = 14, Q = [36028797019389953, 35184372121601, 35184372744193, 35184373006337, 35184373989377, 35184374874113], P = [36028797019488257, 36028797020209153], logDefaultScale = 45>} {
  // CHECK: test_valid_dimension
  func.func @test_valid_dimension(%arg0 : !secret.secret<tensor<1024xi1>> {mgmt.mgmt = #mgmt}) -> (!secret.secret<tensor<1024xi1>> {mgmt.mgmt = #mgmt}) {
    return %arg0 : !secret.secret<tensor<1024xi1>>
  }
}

// -----

#mgmt = #mgmt.mgmt<level = 0, dimension = 2>

// Currently we don't support lowering adds on tensor.insert on into slots of a single ciphertext.

module attributes {ckks.schemeParam = #ckks.scheme_param<logN = 14, Q = [36028797019389953, 35184372121601, 35184372744193, 35184373006337, 35184373989377, 35184374874113], P = [36028797019488257, 36028797020209153], logDefaultScale = 45>} {
  func.func @test_tensor_insert_slot(%arg0 : !secret.secret<tensor<1024xf32>> {mgmt.mgmt = #mgmt}, %arg1 : !secret.secret<f32> {mgmt.mgmt = #mgmt}) -> (!secret.secret<tensor<1024xf32>> {mgmt.mgmt = #mgmt}) {
    %c0 = arith.constant 0 : index
    // expected-error@below {{failed to legalize}}
    %0 = secret.generic(%arg0: !secret.secret<tensor<1024xf32>>, %arg1: !secret.secret<f32>) attrs = {mgmt.mgmt = #mgmt} {
      ^bb0(%ARG0 : tensor<1024xf32>, %ARG1 : f32):
        %1 = tensor.insert %ARG1 into %ARG0[%c0] : tensor<1024xf32>
        secret.yield %1 : tensor<1024xf32>
    } -> !secret.secret<tensor<1024xf32>>
    return %0 : !secret.secret<tensor<1024xf32>>
  }
}
