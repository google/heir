// Tests absorbing constant global memrefs into a generic body during yosys
// optimization. This should optimize the multiplications instead of performing
// a 32-bit generic multiplication.

// RUN: heir-opt --yosys-optimizer %s | FileCheck %s

module attributes {tf_saved_model.semantics} {
  // Use a weight vector with multiple weights to avoid constant folding.
  memref.global "private" constant @__constant_1xi8 : memref<2xi8> = dense<[3, 2]> {alignment = 64 : i64}
  // CHECK: @global_mul_32
  func.func @global_mul_32(%arg0 : !secret.secret<memref<1xi8>>, %weight : i8) -> (!secret.secret<memref<1xi8>>) {
    // Generic 8-bit multiplication
    // CHECK: secret.generic
    // CHECK-COUNT-85: comb.truth_table
    // CHECK: secret.yield
    %0 = secret.generic ins(%arg0 : !secret.secret<memref<1xi8>>) {
    ^bb0(%ARG0 : memref<1xi8>) :
      %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<1xi8>
      affine.parallel (%arg1) = (0) to (1) {
        %1 = affine.load %ARG0[%arg1] : memref<1xi8>
        %3 = arith.muli %1, %weight : i8
        affine.store %3, %alloc_0[%arg1] : memref<1xi8>
      }
      secret.yield %alloc_0 : memref<1xi8>
    } -> !secret.secret<memref<1xi8>>
    // CHECK: return
    return %0 : !secret.secret<memref<1xi8>>
  }
  // CHECK: @global_mul_32_constants
  func.func @global_mul_32_constants(%arg0 : !secret.secret<memref<1xi8>>, %weight : i8) -> (!secret.secret<memref<1xi8>>) {
    // 8-bit multiplication with constant weights
    // CHECK: secret.generic
    // CHECK-COUNT-24: comb.truth_table
    // CHECK: secret.yield
    %4 = memref.get_global @__constant_1xi8 : memref<2xi8>
    %5 = secret.generic ins(%arg0 : !secret.secret<memref<1xi8>>) {
    ^bb0(%ARG0 : memref<1xi8>) :
      %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<1xi8>
      affine.parallel (%arg1) = (0) to (1) {
        %6 = affine.load %ARG0[%arg1] : memref<1xi8>
        %7 = affine.load %4[%arg1] : memref<2xi8>
        %8 = arith.muli %6, %7 : i8
        affine.store %8, %alloc_0[%arg1] : memref<1xi8>
      }
      secret.yield %alloc_0 : memref<1xi8>
    } -> !secret.secret<memref<1xi8>>
    // CHECK: return
    return %5 : !secret.secret<memref<1xi8>>
  }
}
