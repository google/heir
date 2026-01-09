// RUN: heir-opt --annotate-muldepth %s | FileCheck %s

// CHECK: func.func @doctest
// CHECK-SAME: !secret.secret<i32> {secret.mul_depth = 0
// CHECK:   secret.generic
// CHECK-SAME: !secret.secret<i32> {secret.mul_depth = 0
// CHECK:   ^body
// CHECK-NEXT: arith.muli
// CHECK-NEXT: arith.muli
// CHECK-SAME: secret.mul_depth = 1
// CHECK-NEXT: arith.muli
// CHECK-SAME: secret.mul_depth = 1
// CHECK-NEXT: arith.muli
// CHECK-SAME: secret.mul_depth = 2
// CHECK-NEXT: arith.muli
// CHECK-SAME: secret.mul_depth = 3
// CHECK-NEXT: secret.yield

func.func @doctest(%x: !secret.secret<i32>, %y: i32) -> !secret.secret<i32> {
  %0 = secret.generic(%x: !secret.secret<i32>, %y: i32) {
  ^body(%secret_val: i32, %public_val: i32):
    %1 = arith.muli %public_val, %public_val : i32
    %2 = arith.muli %secret_val, %secret_val : i32
    %3 = arith.muli %secret_val, %public_val : i32
    %4 = arith.muli %2, %3 : i32
    %5 = arith.muli %4, %3 : i32
    secret.yield %5 : i32
  } -> !secret.secret<i32>
  return %0 : !secret.secret<i32>
}
