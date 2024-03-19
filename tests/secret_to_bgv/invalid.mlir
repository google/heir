// RUN: heir-opt --split-input-file --secret-to-bgv=="poly-mod-degree=1024" --verify-diagnostics %s | FileCheck %s

// Tests invalid secret types

// expected-error@below {{expected secret types to be tensors with dimension matching ring parameter}}
module {
  func.func @test_not_tensor(%arg0 : !secret.secret<i1>) -> (!secret.secret<i1>) {
    return %arg0 : !secret.secret<i1>
  }
}

// -----

// expected-error@below {{expected secret types to be tensors with dimension matching ring parameter}}
module {
  func.func @test_invalid_dimension(%arg0 : !secret.secret<tensor<1000xi1>>) -> (!secret.secret<tensor<1000xi1>>) {
    return %arg0 : !secret.secret<tensor<1000xi1>>
  }
}

// -----

// CHECK: test_valid_dimension
func.func @test_valid_dimension(%arg0 : !secret.secret<tensor<1024xi1>>) -> (!secret.secret<tensor<1024xi1>>) {
  return %arg0 : !secret.secret<tensor<1024xi1>>
}
