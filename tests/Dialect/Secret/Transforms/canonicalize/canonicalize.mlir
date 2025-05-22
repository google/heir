// RUN: heir-opt --canonicalize %s | FileCheck %s

// CHECK: func @remove_unused_yielded_values
func.func @remove_unused_yielded_values(%arg0: !secret.secret<i32>) -> !secret.secret<i32> {
  %X = arith.constant 7 : i32
  %Y = secret.conceal %X : i32 -> !secret.secret<i32>
  %Z, %UNUSED = secret.generic
    (%Y: !secret.secret<i32>, %arg0: !secret.secret<i32>) {
    ^bb0(%y: i32, %clear_arg0 : i32) :
      %d = arith.addi %clear_arg0, %y: i32
      %unused = arith.addi %y, %y: i32
      // CHECK: secret.yield %[[value:.*]] : i32
      secret.yield %d, %unused : i32, i32
    } -> (!secret.secret<i32>, !secret.secret<i32>)
  return %Z : !secret.secret<i32>
}

// CHECK: func @remove_pass_through_args
func.func @remove_pass_through_args(
// CHECK: %[[arg1:.*]]: !secret.secret<i32>, %[[arg2:.*]]: !secret.secret<i32>
    %arg1 : !secret.secret<i32>, %arg2 : !secret.secret<i32>) -> (!secret.secret<i32>, !secret.secret<i32>) {
  // CHECK: %[[out1:.*]] = secret.generic
  %out1, %out2 = secret.generic
    (%arg1: !secret.secret<i32>, %arg2: !secret.secret<i32>) {
    ^bb0(%x: i32, %y: i32) :
      // CHECK: %[[value:.*]] = arith.addi
      %z = arith.addi %x, %y : i32
      // Only yield one value
      // CHECK: secret.yield %[[value]] : i32
      secret.yield %z, %y : i32, i32
    } -> (!secret.secret<i32>, !secret.secret<i32>)
  // CHECK: return %[[out1]], %[[arg2]]
  return %out1, %out2 : !secret.secret<i32>, !secret.secret<i32>
}

// CHECK: func @hoist_plaintext
// CHECK-SAME: (%arg0: !secret.secret<i16>) -> !secret.secret<i16>
func.func @hoist_plaintext(%arg0: !secret.secret<i16>) -> !secret.secret<i16> {
    // CHECK: arith.constant
    // CHECK: secret.generic
    // CHECK: arith.muli
    // CHECK: secret.yield
    %0 = secret.generic(%arg0 : !secret.secret<i16>) {
    ^bb0(%arg1: i16):
        %c7 = arith.constant 7 : i16
        %0 = arith.muli %arg1, %c7 : i16
         secret.yield %0 : i16
    } -> !secret.secret<i16>
    //CHECK: return %[[S:.*]] : !secret.secret<i16>
    return %0 : !secret.secret<i16>
}

// -----

// CHECK: func @preserve_attrs
func.func @preserve_attrs(%arg0: tensor<1x1024xf32>) -> !secret.secret<tensor<1x1024xf32>> {
  // CHECK: mgmt.init
  // CHECK-SAME: {mgmt.mgmt = #mgmt.mgmt<level = 1>}
    %6 = secret.conceal %arg0 : tensor<1x1024xf32> -> !secret.secret<tensor<1x1024xf32>>
    %7 = secret.generic(%6: !secret.secret<tensor<1x1024xf32>>) {
    ^body(%input0: tensor<1x1024xf32>):
      %10 = mgmt.init %input0 : tensor<1x1024xf32>
      secret.yield %10 : tensor<1x1024xf32>
    } -> (!secret.secret<tensor<1x1024xf32>> {mgmt.mgmt = #mgmt.mgmt<level = 1>})
    //CHECK: return %[[S:.*]] : !secret.secret<tensor<1x1024xf32>>
    return %7 : !secret.secret<tensor<1x1024xf32>>
}
