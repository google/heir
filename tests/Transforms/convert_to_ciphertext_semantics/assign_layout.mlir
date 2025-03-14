// RUN: heir-opt %s --split-input-file --convert-to-ciphertext-semantics=ciphertext-size=32 | FileCheck %s

#vec_layout = affine_map<(d0) -> (d0)>

// CHECK-LABEL: @repeat_vector
func.func @repeat_vector() {
  %0 = secret.generic {
  ^body:
    // CHECK: [[cst:%[^ ]+]] = arith.constant dense<1> : tensor<16xi16>
    %v = arith.constant dense<1> : tensor<16xi16>
    // CHECK: [[encoded:%[^ ]+]] = linalg.concat [[cst]], [[cst]] : tensor<32xi16>
    %0 = tensor_ext.assign_layout %v {layout = #vec_layout, tensor_ext.layout = #vec_layout} : tensor<16xi16>
    secret.yield %0 : tensor<16xi16>
  } -> !secret.secret<tensor<16xi16>>
  return
}

// -----

#row_major = affine_map<(d0, d1) -> (4 * d0 + d1)>

// CHECK-LABEL: @repeat_vector
func.func @diagonal_upsized() {
  %0 = secret.generic {
  ^body:
    // CHECK: [[cst:%[^ ]+]] = arith.constant dense<1> : tensor<4x4xi16>
    %v = arith.constant dense<1> : tensor<4x4xi16>
    // CHECK: [[encoded:%[^ ]+]] = linalg.concat [[cst]], [[cst]] : tensor<32xi16>
    %0 = tensor_ext.assign_layout %v {layout = #row_major, tensor_ext.layout = #row_major} : tensor<4x4xi16>
    secret.yield %0 : tensor<4x4xi16>
  } -> !secret.secret<tensor<4x4xi16>>
  return
}
