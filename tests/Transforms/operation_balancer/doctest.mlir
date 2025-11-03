// RUN: heir-opt --operation-balancer %s | FileCheck %s

// CHECK: func.func @main(%arg0: !secret.secret<i32>, %arg1: !secret.secret<i32>, %arg2: !secret.secret<i32>, %arg3: !secret.secret<i32>, %arg4: !secret.secret<i32>) -> !secret.secret<i32> {
// CHECK-NEXT:    %0 = secret.generic(%arg0: !secret.secret<i32>, %arg1: !secret.secret<i32>, %arg2: !secret.secret<i32>, %arg3: !secret.secret<i32>, %arg4: !secret.secret<i32>) {
// CHECK-NEXT:    ^body(%input0: i32, %input1: i32, %input2: i32, %input3: i32, %input4: i32):
// CHECK-NEXT:      %1 = arith.addi %input0, %input1 : i32
// CHECK-NEXT:      %2 = arith.addi %input2, %input3 : i32
// CHECK-NEXT:      %3 = arith.addi %2, %input4 : i32
// CHECK-NEXT:      %4 = arith.addi %1, %3 : i32
// CHECK-NEXT:      secret.yield %4 : i32
// CHECK-NEXT:   } -> !secret.secret<i32>
// CHECK-NEXT:   return %0 : !secret.secret<i32>
// CHECK-NEXT: }
func.func @main(%arg0 : !secret.secret<i32>, %arg1 : !secret.secret<i32>, %arg2 : !secret.secret<i32>, %arg3 : !secret.secret<i32>, %arg4 : !secret.secret<i32>) -> !secret.secret<i32> {
  %0 = secret.generic(%arg0 : !secret.secret<i32>, %arg1 : !secret.secret<i32>, %arg2 : !secret.secret<i32>, %arg3 : !secret.secret<i32>, %arg4 : !secret.secret<i32>) {
    ^body(%input0 : i32, %input1 : i32, %input2 : i32, %input3 : i32, %input4 : i32):
      %a = arith.addi %input0, %input1 : i32
      %b = arith.addi %a, %input2 : i32
      %c = arith.addi %b, %input3 : i32
      %d = arith.addi %c, %input4 : i32
      secret.yield %d : i32
  } -> !secret.secret<i32>
  return %0 : !secret.secret<i32>
}
