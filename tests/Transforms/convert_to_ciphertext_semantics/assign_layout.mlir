// RUN: heir-opt %s --split-input-file --convert-to-ciphertext-semantics=ciphertext-size=32 | FileCheck %s

#align = #tensor_ext.alignment<in = [16], out = [32]>
#layout = #tensor_ext.layout<map = (d0) -> (d0), alignment = #align>

// Test that a vector of size 16xi16 is replicated to 32xi16.
// CHECK-LABEL: @repeat_vector
func.func @repeat_vector() {
  %0 = secret.generic attrs = {__resattrs = [{tensor_ext.layout = #layout}]} {
  ^body:
    // CHECK: [[cst:%[^ ]+]] = arith.constant dense<1> : tensor<16xi16>
    %v = arith.constant dense<1> : tensor<16xi16>
    // CHECK-COUNT-2: tensor.insert_slice
    // Pass notices the layout is the identity, no need to apply any explicit layout.
    // CHECK-NOT: linalg.generic
    %0 = tensor_ext.assign_layout %v {layout = #layout, tensor_ext.layout = #layout} : tensor<16xi16>
    secret.yield %0 : tensor<16xi16>
  } -> !secret.secret<tensor<16xi16>>
  return
}

// -----

// An edge case of the reassociation grouping logic when expanding a data semantic shape
#align = #tensor_ext.alignment<in = [32], out = [1, 1, 1, 32], insertedDims = [0, 1, 2]>
// The inserted dims do nothing in this edge case test, but the affine map needs
// to account for them.
#layout = #tensor_ext.layout<map = (d0, d1, d2, d3) -> (d3), alignment = #align>

// CHECK-LABEL: @prefix_ones
func.func @prefix_ones() {
  %0 = secret.generic attrs = {__resattrs = [{tensor_ext.layout = #layout}]} {
  ^body:
    // CHECK: [[cst:%[^ ]+]] = arith.constant dense<1>
    %v = arith.constant dense<1> : tensor<32xi16>
    // CHECK: tensor.expand_shape [[cst]] {{\[}}[0, 1, 2, 3]]
    %0 = tensor_ext.assign_layout %v {layout = #layout, tensor_ext.layout = #layout} : tensor<32xi16>
    secret.yield %0 : tensor<32xi16>
  } -> !secret.secret<tensor<32xi16>>
  return
}

// -----

// An edge case of the reassociation grouping logic when expanding a data semantic shape
#align = #tensor_ext.alignment<in = [32], out = [32, 1, 1, 1], insertedDims = [1, 2, 3]>
#layout = #tensor_ext.layout<map = (d0, d1, d2, d3) -> (d0), alignment = #align>

// CHECK-LABEL: @suffix_ones
func.func @suffix_ones() {
  %0 = secret.generic attrs = {__resattrs = [{tensor_ext.layout = #layout}]} {
  ^body:
    // CHECK: [[cst:%[^ ]+]] = arith.constant dense<1>
    %v = arith.constant dense<1> : tensor<32xi16>
    // CHECK: tensor.expand_shape [[cst]] {{\[}}[0, 1, 2, 3]]
    %0 = tensor_ext.assign_layout %v {layout = #layout, tensor_ext.layout = #layout} : tensor<32xi16>
    secret.yield %0 : tensor<32xi16>
  } -> !secret.secret<tensor<32xi16>>
  return
}

// -----

// An edge case of the reassociation grouping logic when expanding a data semantic shape
#align = #tensor_ext.alignment<in = [32], out = [1, 1, 32, 1, 32, 1, 1], insertedDims = [0, 1, 3, 4, 5, 6]>
// The map above replicates to have two axes of size 32, but the ciphertext
// size is still 32, so we have to map the expanded tensor across multiple
// ciphertexts. That expands the scope of this test beyond the grouping of the
// alignment above: we pick 32 ciphertexts each of size 32 and use a trivial
// layout.
#layout = #tensor_ext.layout<map = (d0, d1, d2, d3, d4, d5, d6) -> (d2, d4), alignment = #align>

// CHECK-LABEL: @prefix_and_suffix_ones
func.func @prefix_and_suffix_ones() {
  %0 = secret.generic attrs = {__resattrs = [{tensor_ext.layout = #layout}]} {
  ^body:
    // CHECK: [[cst:%[^ ]+]] = arith.constant dense<1>
    %v = arith.constant dense<1> : tensor<32xi16>
    // CHECK: tensor.expand_shape [[cst]] {{\[}}[0, 1, 2, 3, 4, 5, 6]] output_shape {{\[}}1, 1, 32, 1, 1, 1, 1]
    // CHECK-COUNT-31: tensor.insert_slice
    // CHECK: linalg.generic
    %0 = tensor_ext.assign_layout %v {layout = #layout, tensor_ext.layout = #layout} : tensor<32xi16>
    secret.yield %0 : tensor<32xi16>
  } -> !secret.secret<tensor<32xi16>>
  return
}

// -----

#align = #tensor_ext.alignment<in = [16], out = [32], padding = [16], paddingValue = 0:i16>
#layout = #tensor_ext.layout<map = (d0) -> (d0), alignment = #align>

// CHECK-LABEL: @basic_padding
func.func @basic_padding() {
  %0 = secret.generic attrs = {__resattrs = [{tensor_ext.layout = #layout}]} {
  ^body:
    // CHECK: [[cst:%[^ ]+]] = arith.constant dense<1>
    %v = arith.constant dense<1> : tensor<16xi16>
    // CHECK: tensor.pad [[cst]]
    // No layout materialization needed because the layout map is the identity
    // and the aligned size matches
    // CHECK-NOT: linalg.generic
    %0 = tensor_ext.assign_layout %v {layout = #layout, tensor_ext.layout = #layout} : tensor<16xi16>
    secret.yield %0 : tensor<16xi16>
  } -> !secret.secret<tensor<16xi16>>
  return
}

// -----

#align = #tensor_ext.alignment<in = [4, 4], out = [4, 8]>
#layout = #tensor_ext.layout<map = (d0, d1) -> (4 * d0 + d1), alignment = #align>

// Test that a vector of size 16xi16 is replicated along columns to 4x8 before
// being laid out row major in a single 32-length ciphertext.
// This results in:
//
// ( 1  2  3  4)
// ( 5  6  7  8)
// ( 9 10 11 12)
// (13 14 15 16)
//
// Being laid out as:
//
// (1, 2, 3, 4, 1, 2, 3, 4, 5, 6, 7, 8, 5, 6, 7, 8, ...)
//
// CHECK-LABEL: @column_alignment
func.func @column_alignment() {
  %0 = secret.generic attrs = {__resattrs = [{tensor_ext.layout = #layout}]} {
  ^body:
    // CHECK: [[cst:%[^ ]+]] = arith.constant dense<1> : tensor<4x4xi16>
    %v = arith.constant dense<1> : tensor<4x4xi16>
    // CHECK: tensor.insert_slice
    // CHECK-SAME: [0, 0] [4, 4] [1, 1]
    // CHECK-SAME: tensor<4x4xi16> into tensor<4x8xi16>
    // CHECK: tensor.insert_slice
    // CHECK-SAME: [0, 4] [4, 4] [1, 1]
    // CHECK-SAME: tensor<4x4xi16> into tensor<4x8xi16>
    // CHECK: linalg.generic
    %0 = tensor_ext.assign_layout %v {layout = #layout, tensor_ext.layout = #layout} : tensor<4x4xi16>
    secret.yield %0 : tensor<4x4xi16>
  } -> !secret.secret<tensor<4x4xi16>>
  return
}
