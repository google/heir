// RUN: heir-opt --secret-forget-secrets %s | FileCheck %s

// CHECK-NOT: secret.cast
module {
  func.func private @internal_generic_17345143092562439778(%arg0: i3, %arg1: i3, %arg2: i3, %arg3: i3) -> i3 {
    %0 = arith.subi %arg0, %arg1 : i3
    %1 = arith.muli %0, %arg2 : i3
    %2 = arith.andi %1, %arg3 : i3
    return %2 : i3
  }
  func.func @ops(%arg0: !secret.secret<i3>, %arg1: !secret.secret<i3>, %arg2: !secret.secret<i3>, %arg3: !secret.secret<i3>) -> !secret.secret<i3> {
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %false = arith.constant false
    %0 = secret.cast %arg0 : !secret.secret<i3> to !secret.secret<tensor<3xi1>>
    %1 = secret.cast %arg1 : !secret.secret<i3> to !secret.secret<tensor<3xi1>>
    %2 = secret.cast %arg2 : !secret.secret<i3> to !secret.secret<tensor<3xi1>>
    %3 = secret.cast %arg3 : !secret.secret<i3> to !secret.secret<tensor<3xi1>>
    %4 = secret.generic(%0: !secret.secret<tensor<3xi1>>, %1: !secret.secret<tensor<3xi1>>, %2: !secret.secret<tensor<3xi1>>, %3: !secret.secret<tensor<3xi1>>) {
    ^body(%input0: tensor<3xi1>, %input1: tensor<3xi1>, %input2: tensor<3xi1>, %input3: tensor<3xi1>):
      %6 = tensor.extract %input1[%c0] : tensor<3xi1>
      %7 = tensor.extract %input0[%c0] : tensor<3xi1>
      %8 = comb.truth_table %false, %6, %7 -> 4 : ui8
      %9 = tensor.extract %input1[%c1] : tensor<3xi1>
      %10 = tensor.extract %input0[%c1] : tensor<3xi1>
      %11 = comb.truth_table %9, %10, %8 -> 178 : ui8
      %12 = tensor.extract %input0[%c2] : tensor<3xi1>
      %13 = tensor.extract %input1[%c2] : tensor<3xi1>
      %14 = comb.truth_table %12, %13, %11 -> 105 : ui8
      %15 = tensor.extract %input2[%c0] : tensor<3xi1>
      %16 = comb.truth_table %false, %15, %14 -> 4 : ui8
      %17 = comb.truth_table %false, %10, %9 -> 9 : ui8
      %18 = tensor.extract %input2[%c1] : tensor<3xi1>
      %19 = comb.truth_table %18, %17, %8 -> 144 : ui8
      %20 = comb.truth_table %false, %7, %6 -> 9 : ui8
      %21 = tensor.extract %input2[%c2] : tensor<3xi1>
      %22 = comb.truth_table %19, %21, %20 -> 75 : ui8
      %23 = comb.truth_table %15, %17, %8 -> 144 : ui8
      %24 = comb.truth_table %false, %18, %20 -> 4 : ui8
      %25 = comb.truth_table %22, %24, %23 -> 135 : ui8
      %26 = tensor.extract %input3[%c2] : tensor<3xi1>
      %27 = comb.truth_table %26, %25, %16 -> 96 : ui8
      %28 = tensor.extract %input3[%c0] : tensor<3xi1>
      %29 = comb.truth_table %28, %15, %20 -> 64 : ui8
      %30 = tensor.extract %input3[%c1] : tensor<3xi1>
      %31 = comb.truth_table %30, %24, %23 -> 96 : ui8
      %alloc = tensor.from_elements %29, %31, %27 : tensor<3xi1>
      secret.yield %alloc : tensor<3xi1>
    } -> !secret.secret<tensor<3xi1>>
    %5 = secret.cast %4 : !secret.secret<tensor<3xi1>> to !secret.secret<i3>
    return %5 : !secret.secret<i3>
  }
  func.func private @internal_generic_14758036829565297495(%arg0: i1, %arg1: i1) -> i1 {
    %0 = arith.addi %arg0, %arg1 : i1
    return %0 : i1
  }
  func.func @truth_table(%arg0: !secret.secret<i1>, %arg1: !secret.secret<i1>) -> !secret.secret<i1> {
    %false = arith.constant false
    %0 = secret.generic(%arg0: !secret.secret<i1>, %arg1: !secret.secret<i1>) {
    ^body(%input0: i1, %input1: i1):
      %1 = comb.truth_table %false, %input1, %input0 -> 6 : ui8
      secret.yield %1 : i1
    } -> !secret.secret<i1>
    return %0 : !secret.secret<i1>
  }
}
