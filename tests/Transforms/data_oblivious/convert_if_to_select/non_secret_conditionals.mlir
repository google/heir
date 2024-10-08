// RUN: heir-opt --convert-if-to-select %s | FileCheck %s

// CHECK-LABEL: @non_secret_condition_outside_of_secret_generic_with_secret_tensor
func.func @non_secret_condition_outside_of_secret_generic_with_secret_tensor(%inp: !secret.secret<tensor<16xi16>>, %cond: i1) -> !secret.secret<tensor<16xi16>> {
  // CHECK-NOT: arith.select
  %0 = secret.generic ins(%inp : !secret.secret<tensor<16xi16>>) {
  ^bb0(%arg2: tensor<16xi16>):
    %1 = scf.if %cond -> (tensor<16xi16>) {
      %2 = arith.addi %arg2, %arg2 : tensor<16xi16>
      scf.yield %2 : tensor<16xi16>
    } else {
      scf.yield %arg2 : tensor<16xi16>
    }
    secret.yield %1 : tensor<16xi16>
  } -> !secret.secret<tensor<16xi16>>
  return %0 : !secret.secret<tensor<16xi16>>
}

// CHECK-LABEL: @non_secret_condition_with_secret_tensor
func.func @non_secret_condition_with_secret_tensor(%inp: !secret.secret<tensor<16xi16>>, %cond: i1) -> !secret.secret<tensor<16xi16>> {
  // CHECK-NOT: arith.select
  %0 = secret.generic ins(%inp : !secret.secret<tensor<16xi16>>) {
  ^bb0(%arg2: tensor<16xi16>):
    %1 = scf.if %cond -> (tensor<16xi16>) {
      %2 = arith.addi %arg2, %arg2 : tensor<16xi16>
      scf.yield %2 : tensor<16xi16>
    } else {
      scf.yield %arg2 : tensor<16xi16>
    }
    secret.yield %1 : tensor<16xi16>
  } -> !secret.secret<tensor<16xi16>>
  return %0 : !secret.secret<tensor<16xi16>>
}
