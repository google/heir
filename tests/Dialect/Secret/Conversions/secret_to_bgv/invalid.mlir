// RUN: heir-opt --split-input-file --secret-to-bgv="poly-mod-degree=1024" --verify-diagnostics %s | FileCheck %s

// Tests invalid secret types
#mgmt = #mgmt.mgmt<level = 0, dimension = 2>

module attributes {bgv.schemeParam = #bgv.scheme_param<logN = 14, Q = [67239937, 17179967489, 17180262401, 17180295169, 17180393473, 70368744210433], P = [70368744570881, 70368744701953], plaintextModulus = 65537>} {
  // expected-error@below {{expected secret types to be tensors with last dimension matching ring parameter}}
  func.func @test_invalid_dimension(%arg0 : !secret.secret<tensor<1000xi1>> {mgmt.mgmt = #mgmt}) -> (!secret.secret<tensor<1000xi1>> {mgmt.mgmt = #mgmt}) {
    return %arg0 : !secret.secret<tensor<1000xi1>>
  }
}

// -----

#mgmt = #mgmt.mgmt<level = 0, dimension = 2>

module attributes {bgv.schemeParam = #bgv.scheme_param<logN = 14, Q = [67239937, 17179967489, 17180262401, 17180295169, 17180393473, 70368744210433], P = [70368744570881, 70368744701953], plaintextModulus = 65537>} {
  // CHECK: test_valid_dimension
  func.func @test_valid_dimension(%arg0 : !secret.secret<tensor<1024xi1>> {mgmt.mgmt = #mgmt}) -> (!secret.secret<tensor<1024xi1>> {mgmt.mgmt = #mgmt}) {
    return %arg0 : !secret.secret<tensor<1024xi1>>
  }
}
