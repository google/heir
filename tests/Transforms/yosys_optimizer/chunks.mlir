// RUN: heir-opt --yosys-optimizer="abc-fast=True mode=Boolean" %s | FileCheck %s

// This tests when the RTLIL module contains connections from SigSpec's that are
// neither a chunk of a wire, a full wire, or a bit of a wire, but rather made
// up of multiple disjoint chunks of wires. This module's output connection
// looks like
//   connect { \_out_0 [30:17] \_out_0 [13] \_out_0 [11:10] } { \_out_0 [31] \_out_0 [31] \_out_0 [31] \_out_0 [31] \_out_0 [31] \_out_0 [31] \_out_0 [31] \_out_0 [31] \_out_0 [31] \_out_0 [31] \_out_0 [31] \_out_0 [31] \_out_0 [31] \_out_0 [31] \_out_0 [14] \_out_0 [14] \_out_0 [12] }


// CHECK: @chunks
func.func @chunks(%ARG0: !secret.secret<i8>) -> !secret.secret<i32> {
  // CHECK: secret.cast
  // CHECK: secret.generic
  // CHECK-NOT: arith.extsi
  // CHECK-NOT: arith.subi
  // CHECK-NOT: arith.muli
  // CHECK-NOT: arith.addi
  %1 = secret.generic
      ins(%ARG0: !secret.secret<i8>) {
  ^bb0(%arg1: i8) :
    %c-128_i16_0 = arith.constant -128 : i16
    %c1_i32_1 = arith.constant 1 : i32
    %5 = arith.extsi %arg1 : i8 to i16
    %6 = arith.subi %5, %c-128_i16_0 : i16
    %7 = arith.extsi %6 : i16 to i32
    %8 = arith.addi %7, %c1_i32_1 : i32
    secret.yield %8 : i32
  } -> (!secret.secret<i32>)
  // CHECK: secret.cast
  // CHECK: return
  return %1 : !secret.secret<i32>
}
