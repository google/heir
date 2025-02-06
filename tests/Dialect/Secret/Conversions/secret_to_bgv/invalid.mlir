// RUN: heir-opt --split-input-file --secret-to-bgv="poly-mod-degree=1024" --verify-diagnostics %s | FileCheck %s

// Tests invalid secret types
#mgmt = #mgmt.mgmt<level = 0, dimension = 2>

module {
  // expected-error@below {{expected batched secret types to be tensors with dimension matching ring parameter}}
  func.func @test_invalid_dimension(%arg0 : !secret.secret<tensor<1000xi1>> {mgmt.mgmt = #mgmt}) -> (!secret.secret<tensor<1000xi1>>) {
    return %arg0 : !secret.secret<tensor<1000xi1>>
  }
}

// -----

#mgmt = #mgmt.mgmt<level = 0, dimension = 2>

// CHECK: test_valid_dimension
func.func @test_valid_dimension(%arg0 : !secret.secret<tensor<1024xi1>> {mgmt.mgmt = #mgmt}) -> (!secret.secret<tensor<1024xi1>>) {
  return %arg0 : !secret.secret<tensor<1024xi1>>
}
