// RUN: heir-opt --comb-to-cggi -cse %s | FileCheck %s

// This test was produced by running
//   heir-opt --yosys-optimizer --canonicalize tests/yosys_optimizer/add_one.mlir

module {
  // CHECK: @add_one([[ARG0:%.*]]: [[LWET:tensor<8x!lwe.lwe_ciphertext<.*cleartext_bitwidth = 3.*>>]]) -> [[LWET]]
  func.func @add_one(%arg0: !secret.secret<i8>) -> !secret.secret<i8> {
    // CHECK-NOT: comb
    // CHECK-NOT: secret.generic
    // CHECK-NOT: secret.cast
    // CHECK-COUNT-15: cggi.lut3
    %true = arith.constant true
    %false = arith.constant false
    %c7 = arith.constant 7 : index
    %c6 = arith.constant 6 : index
    %c5 = arith.constant 5 : index
    %c4 = arith.constant 4 : index
    %c3 = arith.constant 3 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %0 = secret.cast %arg0 : !secret.secret<i8> to !secret.secret<tensor<8xi1>>
    %1 = secret.generic ins(%0 : !secret.secret<tensor<8xi1>>) {
    ^bb0(%arg1: tensor<8xi1>):
      %extracted = tensor.extract %arg1[%c0] : tensor<8xi1>
      %3 = comb.truth_table %extracted, %true, %false -> 8 : ui8
      %extracted_0 = tensor.extract %arg1[%c1] : tensor<8xi1>
      %4 = comb.truth_table %3, %extracted_0, %false -> 150 : ui8
      %5 = comb.truth_table %3, %extracted_0, %false -> 23 : ui8
      %extracted_1 = tensor.extract %arg1[%c2] : tensor<8xi1>
      %6 = comb.truth_table %5, %extracted_1, %false -> 43 : ui8
      %extracted_2 = tensor.extract %arg1[%c3] : tensor<8xi1>
      %7 = comb.truth_table %6, %extracted_2, %false -> 43 : ui8
      %extracted_3 = tensor.extract %arg1[%c4] : tensor<8xi1>
      %8 = comb.truth_table %7, %extracted_3, %false -> 43 : ui8
      %extracted_4 = tensor.extract %arg1[%c5] : tensor<8xi1>
      %9 = comb.truth_table %8, %extracted_4, %false -> 43 : ui8
      %extracted_5 = tensor.extract %arg1[%c6] : tensor<8xi1>
      %10 = comb.truth_table %9, %extracted_5, %false -> 105 : ui8
      %11 = comb.truth_table %9, %extracted_5, %false -> 43 : ui8
      %extracted_6 = tensor.extract %arg1[%c7] : tensor<8xi1>
      %12 = comb.truth_table %11, %extracted_6, %false -> 105 : ui8
      %13 = comb.truth_table %extracted, %true, %false -> 6 : ui8
      %14 = comb.truth_table %5, %extracted_1, %false -> 105 : ui8
      %15 = comb.truth_table %6, %extracted_2, %false -> 105 : ui8
      %16 = comb.truth_table %7, %extracted_3, %false -> 105 : ui8
      %17 = comb.truth_table %8, %extracted_4, %false -> 105 : ui8
      %from_elements = tensor.from_elements %12, %10, %17, %16, %15, %14, %4, %13 : tensor<8xi1>
      secret.yield %from_elements : tensor<8xi1>
    } -> !secret.secret<tensor<8xi1>>
    // CHECK: [[CONCAT:%.*]] = tensor.from_elements
    %2 = secret.cast %1 : !secret.secret<tensor<8xi1>> to !secret.secret<i8>
    // CHECK: return [[CONCAT]] : [[LWET]]
    return %2 : !secret.secret<i8>
  }
}
