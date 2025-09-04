// RUN: heir-opt --wrap-generic %s | FileCheck %s

// CHECK: func.func @main(%arg0: !secret.secret<i32>) -> !secret.secret<i32>
// CHECK-NEXT:   %0 = secret.generic(%arg0: !secret.secret<i32>) {
// CHECK-NEXT:   ^body(%input0: i32):
// CHECK-NEXT:     %c100_i32 = arith.constant 100 : i32
// CHECK-NEXT:     %1 = arith.addi %c100_i32, %input0 : i32
// CHECK-NEXT:     secret.yield %1 : i32
// CHECK-NEXT:   } -> !secret.secret<i32>
// CHECK-NEXT:   return %0 : !secret.secret<i32>
// CHECK-NEXT: }
func.func @main(%arg0: i32 {secret.secret}) -> i32 {
  %0 = arith.constant 100 : i32
  %1 = arith.addi %0, %arg0 : i32
  return %1 : i32
}
