// RUN: heir-opt --secret-forget-secrets --split-input-file %s | FileCheck %s

// CHECK-LABEL: test_erase_unused_conceal
func.func @test_erase_unused_conceal(%value : i32) {
  // CHECK-NOT: secret
  %Y = secret.conceal %value : i32 -> !secret.secret<i32>
  func.return
}

// -----

// CHECK-LABEL: test_conceal_then_generic
// CHECK-SAME:     %[[ARG:.*]]: i32
// CHECK-SAME:  ) {
// CHECK:         %[[C7:.*]] = arith.constant 7 : i32
// CHECK:         %[[V0:.*]] = arith.addi %[[C7]], %[[ARG]] : i32
// CHECK:         return
// CHECK:       }
func.func @test_conceal_then_generic(%value : i32) {
  %X = arith.constant 7 : i32
  %Y = secret.conceal %value : i32 -> !secret.secret<i32>
  %Z = secret.generic ins(%X, %Y : i32, !secret.secret<i32>) {
    ^bb0(%x: i32, %y: i32) :
      %d = arith.addi %x, %y: i32
      secret.yield %d : i32
    } -> (!secret.secret<i32>)
  func.return
}

// -----

// CHECK-LABEL: func.func @test_function_signature(
// CHECK-SAME:     %[[ARG:.*]]: i32
// CHECK-SAME:  ) -> i32 {
// CHECK: return %[[ARG]] : i32
func.func @test_function_signature(%Y : !secret.secret<i32>) -> !secret.secret<i32> {
  func.return %Y : !secret.secret<i32>
}

// -----

// CHECK-LABEL: func.func @test_add_two_secrets(
// CHECK-SAME:     %[[S1:.*]]: i32,
// CHECK-SAME:     %[[S2:.*]]: i32
// CHECK-SAME:  ) -> i32 {
// CHECK: %[[V0:.*]] = arith.addi %[[S1]], %[[S2]] : i32
// CHECK: return %[[V0]] : i32
func.func @test_add_two_secrets(
    %s1 : !secret.secret<i32>,
    %s2 : !secret.secret<i32>) -> !secret.secret<i32> {
  %out = secret.generic ins(%s1, %s2 : !secret.secret<i32>, !secret.secret<i32>) {
    ^bb0(%x: i32, %y: i32) :
      %d = arith.addi %x, %y: i32
      secret.yield %d : i32
    } -> (!secret.secret<i32>)
  func.return %out : !secret.secret<i32>
}

// -----

// CHECK-LABEL: func.func @test_compose_generic(
// CHECK-SAME:     %[[S1:.*]]: i32,
// CHECK-SAME:     %[[S2:.*]]: i32
// CHECK-SAME:  ) -> i32 {
// CHECK: %[[V0:.*]] = arith.addi %[[S1]], %[[S2]] : i32
// CHECK: %[[V1:.*]] = arith.muli %[[S1]], %[[V0]] : i32
// CHECK: return %[[V1]] : i32
func.func @test_compose_generic(
    %s1 : !secret.secret<i32>,
    %s2 : !secret.secret<i32>) -> !secret.secret<i32> {
  %0 = secret.generic ins(%s1, %s2 : !secret.secret<i32>, !secret.secret<i32>) {
    ^bb0(%x: i32, %y: i32) :
      %d = arith.addi %x, %y: i32
      secret.yield %d : i32
    } -> (!secret.secret<i32>)
  %1 = secret.generic ins(%s1, %0 : !secret.secret<i32>, !secret.secret<i32>) {
    ^bb0(%x: i32, %y: i32) :
      %d = arith.muli %x, %y: i32
      secret.yield %d : i32
    } -> (!secret.secret<i32>)
  func.return %1 : !secret.secret<i32>
}

func.func @test_convert_call() {
  %0 = arith.constant 7 : i32
  %1 = arith.constant 8 : i32
  %2 = secret.conceal %0 : i32 -> !secret.secret<i32>
  %3 = secret.conceal %1 : i32 -> !secret.secret<i32>
  %4 = func.call @test_compose_generic(%2, %3) : (!secret.secret<i32>, !secret.secret<i32>) -> !secret.secret<i32>
  func.return
}

// -----

// CHECK-LABEL: test_convert_call_2
// CHECK-NOT: secret
func.func @example_fn(
    %s1 : !secret.secret<i32>,
    %s2 : !secret.secret<i32>) -> !secret.secret<i32> {
  func.return %s1 : !secret.secret<i32>
}

func.func @test_convert_call_2() -> i32 {
  %0 = arith.constant 7 : i32
  %1 = arith.constant 8 : i32
  %2 = secret.conceal %0 : i32 -> !secret.secret<i32>
  %3 = secret.conceal %1 : i32 -> !secret.secret<i32>
  %4 = func.call @example_fn(%2, %3) : (!secret.secret<i32>, !secret.secret<i32>) -> !secret.secret<i32>
  %5 = secret.reveal %4 : !secret.secret<i32> -> i32
  func.return %5 : i32
}
