// RUN: heir-opt --secret-to-cggi %s | FileCheck %s

// CHECK: @trivial_loop
// CHECK-NOT: secret
func.func @trivial_loop(%arg0: !secret.secret<tensor<2xi3>>, %arg1: !secret.secret<i3>, %2: !secret.secret<tensor<3xi1>>) -> !secret.secret<i3> {
  %c0 = arith.constant 0 : index
  %0 = affine.for %arg2 = 0 to 2 iter_args(%arg3 = %arg1) -> (!secret.secret<i3>) {
    %1 = secret.generic(%arg0 : !secret.secret<tensor<2xi3>>) {
    ^bb0(%arg4: tensor<2xi3>):
      %4 = tensor.extract %arg4[%c0] : tensor<2xi3>
      secret.yield %4 : i3
    } -> !secret.secret<i3>
    %3 = secret.cast %2 : !secret.secret<tensor<3xi1>> to !secret.secret<i3>
    affine.yield %3 : !secret.secret<i3>
  }
  return %0 : !secret.secret<i3>
}

// CHECK: @sum
// CHECK-NOT: secret
func.func @sum(%arg0: !secret.secret<tensor<2xi3>>) -> !secret.secret<i3> {
  %true = arith.constant true
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c0_i3 = arith.constant 0 : i3
  %0 = secret.conceal %c0_i3 : i3 -> !secret.secret<i3>
  %1 = affine.for %arg1 = 0 to 2 iter_args(%arg2 = %0) -> (!secret.secret<i3>) {
    %2 = secret.cast %arg0 : !secret.secret<tensor<2xi3>> to !secret.secret<tensor<6xi1>>
    %3 = secret.generic(%2 : !secret.secret<tensor<6xi1>>) {
    ^bb0(%arg3: tensor<6xi1>):
      %8 = tensor.extract %arg3[%c1] : tensor<6xi1>
      secret.yield %8 : i1
    } -> !secret.secret<i1>
    %4 = secret.generic(%2 : !secret.secret<tensor<6xi1>>) {
    ^bb0(%arg3: tensor<6xi1>):
      %8 = tensor.extract %arg3[%c2] : tensor<6xi1>
      secret.yield %8 : i1
    } -> !secret.secret<i1>
    %5 = secret.generic(%3: !secret.secret<i1>, %4 : !secret.secret<i1>) {
    ^bb0(%arg3: i1, %arg4: i1):
      %8 = comb.truth_table %true, %arg3, %arg4 -> 1 : ui8
      secret.yield %8 : i1
    } -> !secret.secret<i1>
    %6 = secret.generic(%5: !secret.secret<i1>) {
    ^bb0(%arg3: i1):
      %from_elements = tensor.from_elements %arg3, %arg3, %arg3 : tensor<3xi1>
      secret.yield %from_elements : tensor<3xi1>
    } -> !secret.secret<tensor<3xi1>>
    %7 = secret.cast %6 : !secret.secret<tensor<3xi1>> to !secret.secret<i3>
    affine.yield %7 : !secret.secret<i3>
  }
  return %1 : !secret.secret<i3>
}
