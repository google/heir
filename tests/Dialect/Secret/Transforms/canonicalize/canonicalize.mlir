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

// CHECK: func @no_inputs
func.func @no_inputs() -> !secret.secret<memref<1xf32>> {
  %c0_f32 = arith.constant 0.0 : f32
  %c0 = arith.constant 0 : index
  // CHECK: secret.generic()
  %1 = secret.generic(%c0_f32: f32) {
  ^body(%input0: f32):
    %10 = memref.alloc() : memref<1xf32>
    secret.yield %10 : memref<1xf32>
  } -> (!secret.secret<memref<1xf32>>)
  // CHECK: return
  return %1 : !secret.secret<memref<1xf32>>
}

#layout = #tensor_ext.layout<"{ [] -> [ct, slot] : ct = 0 and 0 <= slot <= 1023 }">
#original_type = #tensor_ext.original_type<originalType = i64, layout = #layout>
func.func private @_assign_layout_465215895469832199(%arg0: i1) -> tensor<1x1024xi1> attributes {client.pack_func = {func_name = "cmux"}} {
  %splat = tensor.splat %arg0 : tensor<1x1024xi1>
  return %splat : tensor<1x1024xi1>
}
// CHECK: func @hoist_call
// CHECK: %[[TRUE:.*]] = arith.constant true
// CHECK: call @_assign_layout_465215895469832199(%[[TRUE]])
// CHECK: secret.generic
// CHECK: return
func.func @hoist_call(%arg0: i64, %arg1: i64, %arg2: !secret.secret<tensor<1x1024xi1>> {tensor_ext.original_type = #tensor_ext.original_type<originalType = i1, layout = #layout>}) -> (!secret.secret<tensor<1x1024xi1>> {tensor_ext.original_type = #original_type}) {
  %true = arith.constant true
  %0 = secret.generic(%arg2: !secret.secret<tensor<1x1024xi1>>) {
  ^body(%input0: tensor<1x1024xi1>):
    %1 = func.call @_assign_layout_465215895469832199(%true) : (i1) -> tensor<1x1024xi1>
    %3 = arith.subi %1, %input0 : tensor<1x1024xi1>
    secret.yield %3 : tensor<1x1024xi1>
  } -> !secret.secret<tensor<1x1024xi1>>
  return %0 : !secret.secret<tensor<1x1024xi1>>
}

// CHECK: func @no_hoist_call
// CHECK: %[[TRUE:.*]] = arith.constant true
// CHECK: secret.generic
// CHECK: call @_nonspeculatable(%[[TRUE]])
// CHECK: return
func.func private @_nonspeculatable(%arg0: i1) -> tensor<1x1024xi1>
func.func @no_hoist_call(%arg0: i64, %arg1: i64, %arg2: !secret.secret<tensor<1x1024xi1>> {tensor_ext.original_type = #tensor_ext.original_type<originalType = i1, layout = #layout>}) -> (!secret.secret<tensor<1x1024xi1>> {tensor_ext.original_type = #original_type}) {
  %true = arith.constant true
  %0 = secret.generic(%arg2: !secret.secret<tensor<1x1024xi1>>) {
  ^body(%input0: tensor<1x1024xi1>):
    %1 = func.call @_nonspeculatable(%true) : (i1) -> tensor<1x1024xi1>
    %3 = arith.subi %1, %input0 : tensor<1x1024xi1>
    secret.yield %3 : tensor<1x1024xi1>
  } -> !secret.secret<tensor<1x1024xi1>>
  return %0 : !secret.secret<tensor<1x1024xi1>>
}
