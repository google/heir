// RUN: heir-opt --secret-distribute-generic --canonicalize --secret-to-cggi -cse %s | FileCheck %s

// This test was produced by running
//   heir-opt --yosys-optimizer --canonicalize tests/yosys_optimizer/add_one.mlir

module {
  // CHECK: func.func @add_one
  // CHECK-NOT: comb
  // CHECK-NOT: secret.generic
  // CHECK-NOT: secret.cast
  // CHECK-COUNT-11: cggi.lut3
  func.func @add_one(%arg0: !secret.secret<i8>) -> !secret.secret<i8> {
    %false = arith.constant false
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    %c5 = arith.constant 5 : index
    %c6 = arith.constant 6 : index
    %c7 = arith.constant 7 : index
    %0 = secret.cast %arg0 : !secret.secret<i8> to !secret.secret<tensor<8xi1>>
    %1 = secret.generic(%0: !secret.secret<tensor<8xi1>>) {
    ^body(%input0: tensor<8xi1>):
      %extracted = tensor.extract %input0[%c1] : tensor<8xi1>
      %extracted_0 = tensor.extract %input0[%c0] : tensor<8xi1>
      %3 = comb.truth_table %false, %extracted, %extracted_0 -> 6 : ui8
      %extracted_1 = tensor.extract %input0[%c2] : tensor<8xi1>
      %4 = comb.truth_table %extracted_1, %extracted, %extracted_0 -> 120 : ui8
      %5 = comb.truth_table %extracted_1, %extracted, %extracted_0 -> 128 : ui8
      %extracted_2 = tensor.extract %input0[%c3] : tensor<8xi1>
      %6 = comb.truth_table %false, %extracted_2, %5 -> 6 : ui8
      %extracted_3 = tensor.extract %input0[%c4] : tensor<8xi1>
      %7 = comb.truth_table %extracted_3, %extracted_2, %5 -> 120 : ui8
      %8 = comb.truth_table %extracted_3, %extracted_2, %5 -> 128 : ui8
      %extracted_4 = tensor.extract %input0[%c5] : tensor<8xi1>
      %9 = comb.truth_table %false, %extracted_4, %8 -> 6 : ui8
      %extracted_5 = tensor.extract %input0[%c6] : tensor<8xi1>
      %10 = comb.truth_table %extracted_5, %extracted_4, %8 -> 120 : ui8
      %11 = comb.truth_table %extracted_5, %extracted_4, %8 -> 128 : ui8
      %extracted_6 = tensor.extract %input0[%c7] : tensor<8xi1>
      %12 = comb.truth_table %false, %extracted_6, %11 -> 6 : ui8
      %13 = comb.truth_table %false, %false, %extracted_0 -> 1 : ui8
      %from_elements = tensor.from_elements %13, %3, %4, %6, %7, %9, %10, %12 : tensor<8xi1>
      secret.yield %from_elements : tensor<8xi1>
    } -> !secret.secret<tensor<8xi1>>
    %2 = secret.cast %1 : !secret.secret<tensor<8xi1>> to !secret.secret<i8>
    return %2 : !secret.secret<i8>
  }
}
