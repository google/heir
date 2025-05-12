// RUN: heir-opt --secret-distribute-generic --secret-to-cggi -cse %s | FileCheck %s

// This test was produced by running
//   heir-opt --yosys-optimizer="mode=Boolean" --canonicalize tests/yosys_optimizer/add_one.mlir

module {
  // CHECK: @add_one
  // CHECK-NOT: comb
  // CHECK-NOT: secret.generic
  // CHECK-NOT: secret.cast
  // CHECK-COUNT-14: cggi
  // CHECK: return
  func.func @add_one(%arg0: !secret.secret<i8>) -> !secret.secret<i8> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    %c5 = arith.constant 5 : index
    %c6 = arith.constant 6 : index
    %c7 = arith.constant 7 : index
    %0 = secret.cast %arg0 : !secret.secret<i8> to !secret.secret<tensor<8xi1>>
    %1 = secret.generic(%0: !secret.secret<tensor<8xi1>>) {
    ^body(%input0: tensor<8xi1>):
      %extracted = tensor.extract %input0[%c0] : tensor<8xi1>
      %3 = comb.inv %extracted : i1
      %extracted_0 = tensor.extract %input0[%c1] : tensor<8xi1>
      %4 = comb.and %extracted, %extracted_0 : i1
      %5 = comb.xor %extracted, %extracted_0 : i1
      %extracted_1 = tensor.extract %input0[%c2] : tensor<8xi1>
      %6 = comb.and %extracted_1, %4 : i1
      %7 = comb.xor %extracted_1, %4 : i1
      %extracted_2 = tensor.extract %input0[%c3] : tensor<8xi1>
      %8 = comb.and %extracted_2, %6 : i1
      %9 = comb.xor %extracted_2, %6 : i1
      %extracted_3 = tensor.extract %input0[%c4] : tensor<8xi1>
      %10 = comb.and %extracted_3, %8 : i1
      %11 = comb.xor %extracted_3, %8 : i1
      %extracted_4 = tensor.extract %input0[%c5] : tensor<8xi1>
      %12 = comb.and %extracted_4, %10 : i1
      %13 = comb.xor %extracted_4, %10 : i1
      %extracted_5 = tensor.extract %input0[%c6] : tensor<8xi1>
      %14 = comb.nand %extracted_5, %12 : i1
      %15 = comb.xor %extracted_5, %12 : i1
      %extracted_6 = tensor.extract %input0[%c7] : tensor<8xi1>
      %16 = comb.xnor %extracted_6, %14 : i1
      %from_elements = tensor.from_elements %3, %5, %7, %9, %11, %13, %15, %16 : tensor<8xi1>
      secret.yield %from_elements : tensor<8xi1>
    } -> !secret.secret<tensor<8xi1>>
    %2 = secret.cast %1 : !secret.secret<tensor<8xi1>> to !secret.secret<i8>
    return %2 : !secret.secret<i8>
  }
}
