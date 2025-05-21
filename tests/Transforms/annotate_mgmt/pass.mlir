// RUN: heir-opt --annotate-mgmt %s | FileCheck %s

// Ensure that mgmt attrs, when present in the function's body already, are
// lifted to the operand and result attrs of the func that contains them.
// CHECK: @lift_to_func
// CHECK-SAME: (%[[arg0:.*]]: !secret.secret<tensor<1024xi16>> {mgmt.mgmt = #mgmt.mgmt<level = 2>}
// CHECK-SAME: -> (!secret.secret<i16> {mgmt.mgmt = #mgmt.mgmt<level = 0>})
func.func @lift_to_func(%arg0: !secret.secret<tensor<1024xi16>>) -> !secret.secret<i16> {
  %c0 = arith.constant {mgmt.mgmt = #mgmt.mgmt<level = 2>} 0 : index
  %c1 = arith.constant {mgmt.mgmt = #mgmt.mgmt<level = 2>} 1 : index
  %0 = secret.generic(%arg0: !secret.secret<tensor<1024xi16>> {mgmt.mgmt = #mgmt.mgmt<level = 2>}) {
  ^body(%input0: tensor<1024xi16>):
    %21 = arith.addi %input0, %input0 {mgmt.mgmt = #mgmt.mgmt<level = 2>} : tensor<1024xi16>
    %22 = tensor_ext.rotate %21, %c1 {mgmt.mgmt = #mgmt.mgmt<level = 2>} : tensor<1024xi16>, index
    %23 = arith.addi %21, %22 {mgmt.mgmt = #mgmt.mgmt<level = 2>} : tensor<1024xi16>
    %24 = mgmt.modreduce %23 {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<1024xi16>
    %extracted = tensor.extract %24[%c0] {mgmt.mgmt = #mgmt.mgmt<level = 1>} : tensor<1024xi16>
    %25 = mgmt.modreduce %extracted {mgmt.mgmt = #mgmt.mgmt<level = 0>} : i16
    secret.yield %25 : i16
  } -> (!secret.secret<i16> {mgmt.mgmt = #mgmt.mgmt<level = 0>})
  return %0 : !secret.secret<i16>
}

// Ensure that mgmt attrs, lift to secret.generic
// CHECK: @lift_to_generic
func.func @lift_to_generic(%arg0: !secret.secret<tensor<1024xi16>>) -> !secret.secret<i16> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // CHECK: secret.generic
  // CHECK-SAME: (%[[arg0:.*]]: !secret.secret<tensor<1024xi16>> {mgmt.mgmt = #mgmt.mgmt<level = 2>})
  %0 = secret.generic(%arg0: !secret.secret<tensor<1024xi16>>) {
  ^body(%input0: tensor<1024xi16>):
    %21 = arith.addi %input0, %input0 {mgmt.mgmt = #mgmt.mgmt<level = 2>} : tensor<1024xi16>
    %22 = tensor_ext.rotate %21, %c1 : tensor<1024xi16>, index
    %23 = arith.addi %21, %22 : tensor<1024xi16>
    %24 = mgmt.modreduce %23 : tensor<1024xi16>
    %extracted = tensor.extract %24[%c0] : tensor<1024xi16>
    %25 = mgmt.modreduce %extracted {mgmt.mgmt = #mgmt.mgmt<level = 0>} : i16
    // CHECK: secret.yield
    // CHECK-NEXT: } -> (!secret.secret<i16> {mgmt.mgmt = #mgmt.mgmt<level = 0>})
    secret.yield %25 : i16
  } -> !secret.secret<i16>
  return %0 : !secret.secret<i16>
}


// Ensure mgmt attrs are inserted from scratch
// CHECK: @annotate_all
// CHECK-SAME: {mgmt.mgmt
// CHECK-SAME: {mgmt.mgmt
func.func @annotate_all(%arg0: !secret.secret<tensor<1024xi16>>) -> !secret.secret<i16> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // CHECK: secret.generic
  // CHECK-SAME: (%[[arg0:.*]]: !secret.secret<tensor<1024xi16>> {mgmt.mgmt
  %0 = secret.generic(%arg0: !secret.secret<tensor<1024xi16>>) {
  // CHECK-NEXT: body
  ^body(%input0: tensor<1024xi16>):
    // CHECK-NEXT: arith.addi
    // CHECK-SAME: {mgmt.mgmt
    %21 = arith.addi %input0, %input0 : tensor<1024xi16>
    // CHECK-NEXT: tensor_ext.rotate
    // CHECK-SAME: {mgmt.mgmt
    %22 = tensor_ext.rotate %21, %c1 : tensor<1024xi16>, index
    // CHECK-NEXT: arith.addi
    // CHECK-SAME: {mgmt.mgmt
    %23 = arith.addi %21, %22 : tensor<1024xi16>
    // CHECK-NEXT: mgmt.modreduce
    // CHECK-SAME: {mgmt.mgmt
    %24 = mgmt.modreduce %23 : tensor<1024xi16>
    // CHECK-NEXT: tensor.extract
    // CHECK-SAME: {mgmt.mgmt
    %extracted = tensor.extract %24[%c0] : tensor<1024xi16>
    // CHECK-NEXT: mgmt.modreduce
    // CHECK-SAME: {mgmt.mgmt
    %25 = mgmt.modreduce %extracted {mgmt.mgmt = #mgmt.mgmt<level = 0>} : i16
    // CHECK: secret.yield
    // CHECK-NEXT: } -> (!secret.secret<i16> {mgmt.mgmt = #mgmt.mgmt<level = 0>})
    secret.yield %25 : i16
  } -> !secret.secret<i16>
  return %0 : !secret.secret<i16>
}

// Ensure mgmt attrs are copied to client helpers
// CHECK: @encrypt_helper
// CHECK-SAME: (%[[arg0:.*]]: tensor<1024xi16>) -> (!secret.secret<tensor<1024xi16>> {mgmt.mgmt
func.func @encrypt_helper(%arg0: tensor<1024xi16>) -> !secret.secret<tensor<1024xi16>> attributes {client_enc_func = {func_name = "annotate_all", index = 0 : i64}} {
  %0 = secret.conceal %arg0 : tensor<1024xi16> -> !secret.secret<tensor<1024xi16>>
  return %0 : !secret.secret<tensor<1024xi16>>
}
// CHECK: @decrypt_helper
// CHECK-SAME: (%[[arg0:.*]]: !secret.secret<tensor<1024xi16>> {mgmt.mgmt
func.func @decrypt_helper(%arg0: !secret.secret<tensor<1024xi16>>) -> tensor<1024xi16> attributes {client_dec_func = {func_name = "annotate_all", index = 0 : i64}} {
  %0 = secret.reveal %arg0 : !secret.secret<tensor<1024xi16>> -> tensor<1024xi16>
  return %0 : tensor<1024xi16>
}
