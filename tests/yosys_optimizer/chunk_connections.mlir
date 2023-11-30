// RUN: heir-opt -yosys-optimizer %s | FileCheck %s

// CHECK-LABEL: @for_loop
func.func @for_loop(%ARG0: !secret.secret<i8>, %ARG1: !secret.secret<i8>) -> !secret.secret<i32> {
  // convert two ARGs
  // CHECK: secret.cast
  // CHECK: secret.cast

  // CHECK: secret.generic
  // CHECK-NOT: arith.extsi
  // CHECK-NOT: arith.subi
  // CHECK-NOT: arith.muli
  // CHECK-NOT: arith.addi
  %1 = secret.generic
      ins(%ARG0, %ARG1: !secret.secret<i8>, !secret.secret<i8>) {
  ^bb0(%arg0: i8, %arg1: i8) :
    %c-128_i16 = arith.constant -128 : i16
    %c0_i32 = arith.constant 0 : i32
    %0 = arith.extsi %arg0 : i8 to i16
    %1 = arith.subi %0, %c-128_i16 : i16
    %2 = arith.extsi %1 : i16 to i32
    %3 = arith.extsi %arg1 : i8 to i32
    %4 = arith.muli %2, %3 : i32
    %5 = arith.addi %c0_i32, %4 : i32
    secret.yield %5 : i32
  } -> (!secret.secret<i32>)

  // CHECK: secret.cast
  // CHECK: return
  return %1 : !secret.secret<i32>
}
