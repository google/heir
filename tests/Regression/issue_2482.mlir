// RUN: heir-opt --secret-to-mod-arith=modulus=17 %s | FileCheck %s

// CHECK: func.func @multi_output
// CHECK-SAME: %[[ARG0:.*]]: ![[TY:.*]]) -> (![[TY]], ![[TY]])
// CHECK: %[[C2:.*]] = arith.constant 2 : i16
// CHECK: %[[EXT:.*]] = arith.extsi %[[C2]] : i16 to i64
// CHECK: %[[ENC:.*]] = mod_arith.encapsulate %[[EXT]] : i64 -> ![[TY]]
// CHECK: %[[MUL:.*]] = mod_arith.mul %[[ARG0]], %[[ENC]] : ![[TY]]
// CHECK: return %[[MUL]], %[[MUL]] : ![[TY]], ![[TY]]
module {
  func.func @multi_output(%arg0: !secret.secret<i16>) -> (!secret.secret<i16>, !secret.secret<i16>) {
    %0 = arith.constant 2 : i16
    %1:2 = "secret.generic"(%arg0) ({
    ^bb0(%arg1: i16):
      %2 = arith.muli %arg1, %0 : i16
      "secret.yield"(%2, %2) : (i16, i16) -> ()
    }) : (!secret.secret<i16>) -> (!secret.secret<i16>, !secret.secret<i16>)
    return %1#0, %1#1 : !secret.secret<i16>, !secret.secret<i16>
  }
}
