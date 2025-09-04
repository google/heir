// RUN: heir-opt --annotate-secretness='verbose=true' %s | FileCheck %s

// CHECK: func.func @mixed_secretness(
// CHECK-SAME: %[[SECRET_ARG:.*]]: !secret.secret<i32> {secret.secret}
// CHECK-SAME: %[[PUBLIC_ARG:.*]]: i32 {secret.public}
// CHECK-SAME: -> (!secret.secret<i32> {secret.secret})
// CHECK:   %[[GENERIC:.*]] = secret.generic(%[[SECRET_ARG]]: !secret.secret<i32>, %[[PUBLIC_ARG]]: i32)
// CHECK-SAME: attrs = {secret.secret}
// CHECK:   ^body(%[[SECRET_VAL:.*]]: i32, %[[PUBLIC_VAL:.*]]: i32):
// CHECK:     %[[PUBLIC_ADD:.*]] = arith.addi %[[PUBLIC_VAL]], %[[PUBLIC_VAL]] {secret.public} : i32
// CHECK:     %[[SECRET_MUL:.*]] = arith.muli %[[SECRET_VAL]], %[[SECRET_VAL]] {secret.secret} : i32
// CHECK:     %[[MIXED_ADD:.*]] = arith.addi %[[SECRET_VAL]], %[[PUBLIC_VAL]] {secret.secret} : i32
// CHECK:     %[[FINAL_ADD:.*]] = arith.addi %[[SECRET_MUL]], %[[MIXED_ADD]] {secret.secret} : i32
// CHECK:     secret.yield %[[FINAL_ADD]] {secret.secret} : i32
// CHECK:   return {secret.secret} %[[GENERIC]]

func.func @mixed_secretness(%secret: !secret.secret<i32>, %public: i32) -> !secret.secret<i32> {
  %0 = secret.generic(%secret: !secret.secret<i32>, %public: i32) {
  ^body(%secret_val: i32, %public_val: i32):
    %1 = arith.addi %public_val, %public_val : i32          // public + public = public
    %2 = arith.muli %secret_val, %secret_val : i32          // secret * secret = secret
    %3 = arith.addi %secret_val, %public_val : i32          // secret + public = secret
    %4 = arith.addi %2, %3 : i32                           // secret + secret = secret
    secret.yield %4 : i32
  } -> !secret.secret<i32>
  return %0 : !secret.secret<i32>
}
