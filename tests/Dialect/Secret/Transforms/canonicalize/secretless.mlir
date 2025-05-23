// RUN: heir-opt --canonicalize --split-input-file %s | FileCheck %s

// CHECK: func @func
// CHECK-SAME: (%arg0: !secret.secret<i16>) -> !secret.secret<i16>
func.func @func(%arg0: !secret.secret<i16>) -> !secret.secret<i16> {
    // CHECK: %[[c1:.*]] = arith.constant 1 : i16
    // CHECK: %[[v0:.*]] = secret.conceal %[[c1]] : i16 -> !secret.secret<i16>
    // CHECK: return %[[v0]] : !secret.secret<i16>
    %0 = secret.generic(%arg0 : !secret.secret<i16>) {
    ^bb0(%arg1: i16):
        %c1_i16 = arith.constant 1 : i16
        secret.yield %c1_i16 : i16
    } -> !secret.secret<i16>
    return %0 : !secret.secret<i16>
}

// -----

// CHECK: func @constant_false
// CHECK-SAME: () -> !secret.secret<i2>
func.func @constant_false() -> !secret.secret<i2> {
  // CHECK: %[[cst:.*]] = arith.constant dense<false> : tensor<2xi1>
  // CHECK: %[[v0:.*]] = secret.conceal %[[cst]] : tensor<2xi1> -> !secret.secret<tensor<2xi1>>
  // CHECK: %[[v1:.*]] = secret.cast %[[v0]] : !secret.secret<tensor<2xi1>> to !secret.secret<i2>
  // CHECK: return %[[v1]] : !secret.secret<i2>
  %false = arith.constant false
  %0 = secret.generic(%false : i1) {
  ^bb0(%arg1: i1):
      %1 = tensor.from_elements %arg1, %arg1 : tensor<2xi1>
      secret.yield %1 : tensor<2xi1>
  } -> !secret.secret<tensor<2xi1>>
  %1 = secret.cast %0 : !secret.secret<tensor<2xi1>> to !secret.secret<i2>
  return %1 : !secret.secret<i2>
}

// -----

// CHECK: func @no_conceal
// CHECK-SAME: (%[[arg0:.*]]: !secret.secret<i2>) -> !secret.secret<i2>
func.func @no_conceal(%arg0: !secret.secret<i2>) -> !secret.secret<i2> {
  // CHECK-NEXT: %[[cst:.*]] = arith.constant -1 : i2
  // CHECK-NEXT: %[[v1:.*]] = secret.generic(%[[arg0]]: !secret.secret<i2>)
  // CHECK-NEXT: ^body(%[[input0:.*]]: i2):
  // CHECK-NEXT: %[[v2:.*]] = arith.muli %[[input0]], %[[cst]] : i2
  // CHECK-NEXT: secret.yield %[[v2]] : i2
  // CHECK: return %[[v1]] : !secret.secret<i2>
  %false = arith.constant 3 : i2
  %secret_false = secret.conceal %false : i2 -> !secret.secret<i2>
  %0 = secret.generic(%arg0 : !secret.secret<i2>, %secret_false : !secret.secret<i2>) {
  ^bb0(%arg1: i2, %arg2: i2):
      %1 = arith.muli %arg1, %arg2 : i2
      secret.yield %1 : i2
  } -> !secret.secret<i2>
  return %0 : !secret.secret<i2>
}

// -----

// CHECK: func @no_conceal_from_elements
// CHECK-SAME: (%[[pt:.*]]: i1) -> !secret.secret<i1>
func.func @no_conceal_from_elements(%pt: i1) -> !secret.secret<i1> {
  // CHECK: %[[v0:.*]] = secret.conceal %[[pt]] : i1 -> !secret.secret<i1>
  // CHECK: return %[[v0]] : !secret.secret<i1>
  %false = arith.constant false
  %c0 = arith.constant 0 : index
  %0 = secret.generic(%pt : i1) {
  ^bb0(%arg1: i1):
      %1 = tensor.from_elements %arg1, %arg1 : tensor<2xi1>
      %2 = tensor.extract %1[%c0] : tensor<2xi1>
      secret.yield %2 : i1
  } -> !secret.secret<i1>
  return %0 : !secret.secret<i1>
}
