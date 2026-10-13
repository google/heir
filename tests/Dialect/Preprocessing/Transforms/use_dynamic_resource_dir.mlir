// RUN: heir-opt --preprocessing-use-dynamic-resource-dir %s | FileCheck %s

// Loading functions and their callers gain a trailing directory argument; the
// loads read it and calls forward it.

// CHECK: func.func private @layout(
// CHECK-SAME:    %[[IN:.*]]: tensor<4xf32>,
// CHECK-SAME:    %[[DIR:.*]]: !preprocessing.resource_dir) -> tensor<4xf32>
// CHECK:         %[[EMPTY:.*]] = tensor.empty() : tensor<4xf32>
// CHECK:         %[[WEIGHTS:.*]] = preprocessing.load_resource "weights.bin" from %[[DIR]] into %[[EMPTY]] : (!preprocessing.resource_dir, tensor<4xf32>) -> tensor<4xf32>
// CHECK:         %[[SUM:.*]] = arith.addf %[[IN]], %[[WEIGHTS]] : tensor<4xf32>
// CHECK:         return %[[SUM]] : tensor<4xf32>
func.func private @layout(%input: tensor<4xf32>) -> tensor<4xf32> {
  %empty = tensor.empty() : tensor<4xf32>
  %weights = preprocessing.load_resource "weights.bin" into %empty : (tensor<4xf32>) -> tensor<4xf32>
  %sum = arith.addf %input, %weights : tensor<4xf32>
  return %sum : tensor<4xf32>
}

// CHECK: func.func @preprocess(
// CHECK-SAME:    %[[IN:.*]]: tensor<4xf32>,
// CHECK-SAME:    %[[DIR:.*]]: !preprocessing.resource_dir) -> tensor<4xf32>
// CHECK:         %[[RES:.*]] = {{(func\.)?}}call @layout(%[[IN]], %[[DIR]]) : (tensor<4xf32>, !preprocessing.resource_dir) -> tensor<4xf32>
// CHECK:         return %[[RES]] : tensor<4xf32>
func.func @preprocess(%input: tensor<4xf32>) -> tensor<4xf32> {
  %result = func.call @layout(%input) : (tensor<4xf32>) -> tensor<4xf32>
  return %result : tensor<4xf32>
}

// A function that loads nothing is unchanged.
// CHECK: func.func @evaluate(
// CHECK-SAME:    %[[IN:.*]]: tensor<4xf32>) -> tensor<4xf32>
// CHECK-NOT:     !preprocessing.resource_dir
func.func @evaluate(%input: tensor<4xf32>) -> tensor<4xf32> {
  return %input : tensor<4xf32>
}

// Transitive callers across multiple levels of call graph:
// CHECK: func.func private @leaf_loader(
// CHECK-SAME:    %[[DIR:.*]]: !preprocessing.resource_dir) -> tensor<4xi32>
// CHECK:         preprocessing.load_resource "leaf.bin" from %[[DIR]] into
func.func private @leaf_loader() -> tensor<4xi32> {
  %empty = tensor.empty() : tensor<4xi32>
  %res = preprocessing.load_resource "leaf.bin" into %empty : (tensor<4xi32>) -> tensor<4xi32>
  return %res : tensor<4xi32>
}

// CHECK: func.func private @mid_caller(
// CHECK-SAME:    %[[DIR:.*]]: !preprocessing.resource_dir) -> tensor<4xi32>
// CHECK:         {{(func\.)?}}call @leaf_loader(%[[DIR]])
func.func private @mid_caller() -> tensor<4xi32> {
  %res = func.call @leaf_loader() : () -> tensor<4xi32>
  return %res : tensor<4xi32>
}

// CHECK: func.func @top_caller(
// CHECK-SAME:    %[[DIR:.*]]: !preprocessing.resource_dir) -> tensor<4xi32>
// CHECK:         {{(func\.)?}}call @mid_caller(%[[DIR]])
func.func @top_caller() -> tensor<4xi32> {
  %res = func.call @mid_caller() : () -> tensor<4xi32>
  return %res : tensor<4xi32>
}

// Multiple calls to the same or different loaders from one function:
// CHECK: func.func @multi_caller(
// CHECK-SAME:    %[[DIR:.*]]: !preprocessing.resource_dir) -> (tensor<4xi32>, tensor<4xi32>)
// CHECK:         %[[R1:.*]] = {{(func\.)?}}call @leaf_loader(%[[DIR]])
// CHECK:         %[[R2:.*]] = {{(func\.)?}}call @leaf_loader(%[[DIR]])
// CHECK:         return %[[R1]], %[[R2]]
func.func @multi_caller() -> (tensor<4xi32>, tensor<4xi32>) {
  %0 = func.call @leaf_loader() : () -> tensor<4xi32>
  %1 = func.call @leaf_loader() : () -> tensor<4xi32>
  return %0, %1 : tensor<4xi32>, tensor<4xi32>
}

// Loads that already have a directory operand are preserved without adding duplicates:
// CHECK: func.func @already_has_dir(
// CHECK-SAME:    %[[PRE_DIR:.*]]: !preprocessing.resource_dir) -> tensor<4xi32>
// CHECK-NOT:     !preprocessing.resource_dir, !preprocessing.resource_dir
// CHECK:         preprocessing.load_resource "manual.bin" from %[[PRE_DIR]] into
func.func @already_has_dir(%dir: !preprocessing.resource_dir) -> tensor<4xi32> {
  %empty = tensor.empty() : tensor<4xi32>
  %res = preprocessing.load_resource "manual.bin" from %dir into %empty : (!preprocessing.resource_dir, tensor<4xi32>) -> tensor<4xi32>
  return %res : tensor<4xi32>
}

// Self-recursive function that loads resources.
// CHECK: func.func @recursive(
// CHECK-SAME:    %[[ARG:.*]]: i32,
// CHECK-SAME:    %[[DIR:.*]]: !preprocessing.resource_dir) -> tensor<4xi32>
// CHECK:         %[[COND:.*]] = arith.cmpi sgt, %[[ARG]]
// CHECK:         scf.if %[[COND]]
// CHECK:           %[[DEC:.*]] = arith.subi %[[ARG]]
// CHECK:           {{(func\.)?}}call @recursive(%[[DEC]], %[[DIR]])
// CHECK:         %[[EMPTY:.*]] = tensor.empty() : tensor<4xi32>
// CHECK:         preprocessing.load_resource "rec.bin" from %[[DIR]] into %[[EMPTY]]
func.func @recursive(%arg: i32) -> tensor<4xi32> {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %cond = arith.cmpi sgt, %arg, %c0 : i32
  scf.if %cond {
    %dec = arith.subi %arg, %c1 : i32
    %rec = func.call @recursive(%dec) : (i32) -> tensor<4xi32>
  }
  %empty = tensor.empty() : tensor<4xi32>
  %res = preprocessing.load_resource "rec.bin" into %empty : (tensor<4xi32>) -> tensor<4xi32>
  return %res : tensor<4xi32>
}

// Loading and calling inside nested regions (scf.if and scf.for).
// CHECK: func.func @nested_regions(
// CHECK-SAME:    %[[COND:.*]]: i1,
// CHECK-SAME:    %[[DIR:.*]]: !preprocessing.resource_dir) -> tensor<4xi32>
// CHECK:         scf.if %[[COND]]
// CHECK:           scf.for
// CHECK:             preprocessing.load_resource "nested.bin" from %[[DIR]] into
// CHECK:             {{(func\.)?}}call @leaf_loader(%[[DIR]])
func.func @nested_regions(%cond: i1) -> tensor<4xi32> {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  %empty = tensor.empty() : tensor<4xi32>
  %res = scf.if %cond -> (tensor<4xi32>) {
    %loop = scf.for %i = %c0 to %c4 step %c1 iter_args(%acc = %empty) -> (tensor<4xi32>) {
      %l = preprocessing.load_resource "nested.bin" into %acc : (tensor<4xi32>) -> tensor<4xi32>
      %c = func.call @leaf_loader() : () -> tensor<4xi32>
      scf.yield %c : tensor<4xi32>
    }
    scf.yield %loop : tensor<4xi32>
  } else {
    scf.yield %empty : tensor<4xi32>
  }
  return %res : tensor<4xi32>
}

// Function with pre-existing directory parameter: verifies no duplicate parameter is added.
// CHECK: func.func @existing_dir(
// CHECK-SAME:    %[[DIR:.*]]: !preprocessing.resource_dir) -> tensor<4xi32>
// CHECK-NOT:     !preprocessing.resource_dir, !preprocessing.resource_dir
// CHECK:         preprocessing.load_resource "existing.bin" from %[[DIR]] into
func.func @existing_dir(%dir: !preprocessing.resource_dir) -> tensor<4xi32> {
  %empty = tensor.empty() : tensor<4xi32>
  %res = preprocessing.load_resource "existing.bin" into %empty : (tensor<4xi32>) -> tensor<4xi32>
  return %res : tensor<4xi32>
}

// Caller that already has a directory parameter and calls existing_dir:
// CHECK: func.func @caller_already_has_dir(
// CHECK-SAME:    %[[DIR:.*]]: !preprocessing.resource_dir) -> tensor<4xi32>
// CHECK-NOT:     !preprocessing.resource_dir, !preprocessing.resource_dir
// CHECK:         {{(func\.)?}}call @existing_dir(%[[DIR]])
func.func @caller_already_has_dir(%dir: !preprocessing.resource_dir) -> tensor<4xi32> {
  %res = func.call @existing_dir(%dir) : (!preprocessing.resource_dir) -> tensor<4xi32>
  return %res : tensor<4xi32>
}

// Preservation of arg_attrs and res_attrs on func.func and func.call.
// CHECK: func.func private @callee_with_attrs(
// CHECK-SAME:    %[[ARG:.*]]: tensor<4xi32> {test.arg_attr = 1 : i32},
// CHECK-SAME:    %[[DIR:.*]]: !preprocessing.resource_dir) -> (tensor<4xi32> {test.res_attr = 2 : i32})
// CHECK:         preprocessing.load_resource "attrs.bin" from %[[DIR]] into
func.func private @callee_with_attrs(%arg: tensor<4xi32> {test.arg_attr = 1 : i32}) -> (tensor<4xi32> {test.res_attr = 2 : i32}) {
  %empty = tensor.empty() : tensor<4xi32>
  %res = preprocessing.load_resource "attrs.bin" into %empty : (tensor<4xi32>) -> tensor<4xi32>
  return %res : tensor<4xi32>
}

// CHECK: func.func @caller_with_attrs(
// CHECK-SAME:    %[[IN:.*]]: tensor<4xi32>,
// CHECK-SAME:    %[[DIR:.*]]: !preprocessing.resource_dir) -> tensor<4xi32>
// CHECK:         {{(func\.)?}}call @callee_with_attrs(%[[IN]], %[[DIR]]) <arg_attrs = [{test.call_arg_attr = 3 : i32}, {}]> : (tensor<4xi32>, !preprocessing.resource_dir) -> tensor<4xi32>
func.func @caller_with_attrs(%input: tensor<4xi32>) -> tensor<4xi32> {
  %res = func.call @callee_with_attrs(%input) <arg_attrs = [{test.call_arg_attr = 3 : i32}]> : (tensor<4xi32>) -> tensor<4xi32>
  return %res : tensor<4xi32>
}
