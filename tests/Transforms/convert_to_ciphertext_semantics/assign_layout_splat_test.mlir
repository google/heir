// RUN: heir-opt %s --convert-to-ciphertext-semantics=ciphertext-size=32 --split-input-file | FileCheck %s

// Splat tensor with dense layout.
// Input tensor<32xi16> with splat 1, dense layout to 1x32 ciphertext.
// CHECK: @test_splat_tensor_dense
#layout_dense = #tensor_ext.layout<"{ [i0] -> [ct, slot] : ct = 0 and 0 <= i0 <= 31 and slot = i0 }">
module {
  func.func @test_splat_tensor_dense() -> (!secret.secret<tensor<32xi16>> {tensor_ext.layout = #layout_dense}) {
    %cst = arith.constant dense<1> : tensor<32xi16>
    // CHECK-NOT: func.call
    // CHECK: arith.constant dense<1> : tensor<1x32xi16>
    %0 = secret.generic() {
      %1 = tensor_ext.assign_layout %cst {layout = #layout_dense, tensor_ext.layout = #layout_dense} : tensor<32xi16>
      secret.yield %1 : tensor<32xi16>
    } -> (!secret.secret<tensor<32xi16>> {tensor_ext.layout = #layout_dense})
    // CHECK: return
    return %0 : !secret.secret<tensor<32xi16>>
  }
}

// -----

// Splat tensor with non-dense layout.
// Input tensor<16xi16> with splat 1, layout only covers first 16 slots.
// This should be OUTLINED because it uses a loop nest.
// CHECK: func.func private @_assign_layout
// CHECK: scf.for
// CHECK: @test_splat_tensor_not_dense
#layout_not_dense = #tensor_ext.layout<"{ [i0] -> [ct, slot] : ct = 0 and 0 <= i0 <= 15 and slot = i0 }">
module {
  func.func @test_splat_tensor_not_dense() -> (!secret.secret<tensor<16xi16>> {tensor_ext.layout = #layout_not_dense}) {
    %cst = arith.constant dense<1> : tensor<16xi16>
    // CHECK: func.call @_assign_layout
    %0 = secret.generic() {
      %1 = tensor_ext.assign_layout %cst {layout = #layout_not_dense, tensor_ext.layout = #layout_not_dense} : tensor<16xi16>
      secret.yield %1 : tensor<16xi16>
    } -> (!secret.secret<tensor<16xi16>> {tensor_ext.layout = #layout_not_dense})
    // CHECK: return
    return %0 : !secret.secret<tensor<16xi16>>
  }
}

// -----

// Scalar constant with dense layout.
// This should be INLINED as a splat.
// CHECK: @test_scalar_dense
#layout_scalar = #tensor_ext.layout<"{ [] -> [ct, slot] : ct = 0 and 0 <= slot <= 31 }">
module {
  func.func @test_scalar_dense(%arg0 : !secret.secret<i16> {tensor_ext.layout = #layout_scalar}) -> (!secret.secret<i16> {tensor_ext.layout = #layout_scalar}) {
    %0 = secret.generic(%arg0 : !secret.secret<i16> {tensor_ext.layout = #layout_scalar}) {
    ^body(%input: i16):
      %c2 = arith.constant 2 : i16
      // CHECK-NOT: func.call
      // CHECK: arith.constant dense<2> : tensor<1x32xi16>
      %1 = tensor_ext.assign_layout %c2 {layout = #layout_scalar, tensor_ext.layout = #layout_scalar} : i16
      %2 = arith.muli %input, %1 {tensor_ext.layout = #layout_scalar} : i16
      secret.yield %2 : i16
    } -> (!secret.secret<i16> {tensor_ext.layout = #layout_scalar})
    // CHECK: return
    return %0 : !secret.secret<i16>
  }
}

// -----

// Non-constant scalar with dense layout.
// This should be INLINED as a splat.
// CHECK: @test_scalar_non_constant
#layout_scalar = #tensor_ext.layout<"{ [] -> [ct, slot] : ct = 0 and 0 <= slot <= 31 }">
module {
  func.func @test_scalar_non_constant(%arg0 : i16) -> (!secret.secret<i16> {tensor_ext.layout = #layout_scalar}) {
    // CHECK: secret.generic
    // CHECK-SAME: %[[arg0:.*]]: i16
    // CHECK-NEXT: (%[[input0:.*]]: i16)
    // CHECK-NOT: func.call
    // CHECK: tensor.splat %[[input0]] : tensor<1x32xi16>
    %0 = secret.generic(%arg0 : i16) {
    ^body(%input: i16):
      %1 = tensor_ext.assign_layout %input {layout = #layout_scalar, tensor_ext.layout = #layout_scalar} : i16
      secret.yield %1 : i16
    } -> (!secret.secret<i16> {tensor_ext.layout = #layout_scalar})
    return %0 : !secret.secret<i16>
  }
}

// -----

// Splat DenseResource with dense layout.
// CHECK: @test_dense_resource_dense
#layout_dense_res = #tensor_ext.layout<"{ [i0] -> [ct, slot] : ct = 0 and 0 <= i0 <= 3 and 0 <= slot <= 31 and slot % 4 = i0 }">
module {
  func.func @test_dense_resource_dense() -> (!secret.secret<tensor<4xf32>> {tensor_ext.layout = #layout_dense_res}) {
    %cst = arith.constant dense_resource<resource1> : tensor<4xf32>
    // CHECK: func.call
    // CHECK: return
    %0 = secret.generic() {
      %1 = tensor_ext.assign_layout %cst {layout = #layout_dense_res, tensor_ext.layout = #layout_dense_res} : tensor<4xf32>
      secret.yield %1 : tensor<4xf32>
    } -> (!secret.secret<tensor<4xf32>> {tensor_ext.layout = #layout_dense_res})
    return %0 : !secret.secret<tensor<4xf32>>
  }
}

{-#
  dialect_resources: {
    builtin: {
      resource1: "0x40000000CDCC8C3FCDCC0C403333534000000000"
    }
  }
#-}

// -----

// Dense but not constant
// CHECK: func.func private @_assign_layout
// CHECK: scf.for
// CHECK: @test_dense_but_not_constant
#layout_not_dense = #tensor_ext.layout<"{ [i0] -> [ct, slot] : ct = 0 and 0 <= i0 <= 15 and 0 <= slot <= 31 and slot % 16 = i0 }">
module {
  func.func @test_dense_but_not_constant() -> (!secret.secret<tensor<16xi16>> {tensor_ext.layout = #layout_not_dense}) {
    %cst = arith.constant dense<[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]> : tensor<16xi16>
    // CHECK: func.call @_assign_layout
    %0 = secret.generic() {
      %1 = tensor_ext.assign_layout %cst {layout = #layout_not_dense, tensor_ext.layout = #layout_not_dense} : tensor<16xi16>
      secret.yield %1 : tensor<16xi16>
    } -> (!secret.secret<tensor<16xi16>> {tensor_ext.layout = #layout_not_dense})
    // CHECK: return
    return %0 : !secret.secret<tensor<16xi16>>
  }
}
