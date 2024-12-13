// RUN: heir-opt --split-input-file --secret-to-ckks="poly-mod-degree=1024" --verify-diagnostics %s | FileCheck %s

// Tests invalid secret types

// expected-warning@below {{expected secret types to be tensors with dimension matching ring parameter, pass will not pack tensors into ciphertext SIMD slots}}
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

// -----

// Currently we don't support lowering adds on tensors of ciphertexts - the
// lowering must implement a loop of add operations on each element.

// expected-warning@below {{expected secret types to be tensors with dimension matching ring parameter, pass will not pack tensors into ciphertext SIMD slots}}
module {
  func.func @test_add_tensor_not_packed(%arg0 : !secret.secret<tensor<1023xf32>>) -> (!secret.secret<tensor<1023xf32>>) {
    // expected-error@below {{failed to legalize}}
    %0 = secret.generic ins(%arg0 :  !secret.secret<tensor<1023xf32>>) {
      ^bb0(%ARG0 : tensor<1023xf32>):
        %1 = arith.addf %ARG0, %ARG0 : tensor<1023xf32>
        secret.yield %1 : tensor<1023xf32>
    } -> !secret.secret<tensor<1023xf32>>
    return %0 : !secret.secret<tensor<1023xf32>>
  }
}
// -----

#mgmt = #mgmt.mgmt<level = 0, dimension = 2>

// Currently we don't support lowering adds on tensor.insert on into slots of a single ciphertext.

module {
  func.func @test_tensor_insert_slot(%arg0 : !secret.secret<tensor<1024xf32>> {mgmt.mgmt = #mgmt}, %arg1 : !secret.secret<f32> {mgmt.mgmt = #mgmt}) -> (!secret.secret<tensor<1024xf32>>) {
    %c0 = arith.constant 0 : index
    // expected-error@below {{failed to legalize}}
    %0 = secret.generic ins(%arg0, %arg1 :  !secret.secret<tensor<1024xf32>>, !secret.secret<f32>) attrs = {mgmt.mgmt = #mgmt} {
      ^bb0(%ARG0 : tensor<1024xf32>, %ARG1 : f32):
        %1 = tensor.insert %ARG1 into %ARG0[%c0] : tensor<1024xf32>
        secret.yield %1 : tensor<1024xf32>
    } -> !secret.secret<tensor<1024xf32>>
    return %0 : !secret.secret<tensor<1024xf32>>
  }
}
