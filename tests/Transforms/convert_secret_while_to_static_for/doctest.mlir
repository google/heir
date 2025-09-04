// RUN: heir-opt --convert-secret-while-to-static-for %s | FileCheck %s

// CHECK: func.func @main
// CHECK:      %[[C100:.*]] = arith.constant 100 : i16
// CHECK:      %[[GENERIC:.*]] = secret.generic
// CHECK:      ^body(%[[INPUT:.*]]: i16):
// CHECK-NEXT:   %[[FOR:.*]] = affine.for %{{.*}} = 0 to 16 iter_args(%[[ARG:.*]] = %[[INPUT]]) -> (i16) {
// CHECK-NEXT:     %[[CMP:.*]] = arith.cmpi sgt, %[[ARG]], %[[C100]] : i16
// CHECK-NEXT:     %[[IF:.*]] = scf.if %[[CMP]] -> (i16) {
// CHECK-NEXT:       %[[MUL:.*]] = arith.muli %[[ARG]], %[[ARG]] : i16
// CHECK-NEXT:       scf.yield %[[MUL]] : i16
// CHECK-NEXT:     } else {
// CHECK-NEXT:       scf.yield %[[ARG]] : i16
// CHECK-NEXT:     }
// CHECK-NEXT:     affine.yield %[[IF]] : i16
// CHECK-NEXT:   }
// CHECK-NEXT:   secret.yield %[[FOR]] : i16
// CHECK:      return

func.func @main(%secretInput: !secret.secret<i16>) -> !secret.secret<i16> {
  %c100 = arith.constant 100 : i16
  %0 = secret.generic(%secretInput: !secret.secret<i16>) {
  ^bb0(%input: i16):
    %1 = scf.while (%arg1 = %input) : (i16) -> i16 {
      %2 = arith.cmpi sgt, %arg1, %c100 : i16
      scf.condition(%2) %arg1 : i16
    } do {
    ^bb0(%arg1: i16):
      %3 = arith.muli %arg1, %arg1 : i16
      scf.yield %3 : i16
    } attributes {max_iter = 16 : i64}
    secret.yield %1 : i16
  } -> !secret.secret<i16>
  return %0 : !secret.secret<i16>
}
