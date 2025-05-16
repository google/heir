// RUN: heir-opt --full-loop-unroll --canonicalize --convert-tensor-to-scalars --canonicalize --cse %s | FileCheck %s

module {
  // CHECK: func.func @main
  // CHECK-SAME: (%[[arg0:.*]]: i8)
  // CHECK-SAME: -> (i32, i32, i32)
  func.func @main(%arg0: tensor<1x1xi8>) -> tensor<1x3xi32> {
    %cst = arith.constant dense<[[9, 54, 57]]> : tensor<1x3xi8>
    // CHECK-DAG: arith.constant 1000 : i32
    // CHECK-DAG: arith.constant 2000 : i32
    // CHECK-DAG: arith.constant 5438 : i32
    %cst_0 = arith.constant dense<[[1000, 2000, 5438]]> : tensor<1x3xi32>
    %c0 = arith.constant 0 : index
    %0 = affine.for %arg1 = 0 to 3 iter_args(%arg2 = %cst_0) -> (tensor<1x3xi32>) {
      %extracted = tensor.extract %arg0[%c0, %c0] : tensor<1x1xi8>
      %extracted_1 = tensor.extract %cst[%c0, %arg1] : tensor<1x3xi8>
      %extracted_2 = tensor.extract %arg2[%c0, %arg1] : tensor<1x3xi32>
      %1 = arith.extsi %extracted : i8 to i32
      %2 = arith.extsi %extracted_1 : i8 to i32
      %3 = arith.muli %1, %2 : i32
      %4 = arith.addi %extracted_2, %3 : i32
      %inserted = tensor.insert %4 into %arg2[%c0, %arg1] : tensor<1x3xi32>
      affine.yield %inserted : tensor<1x3xi32>
    }
    return %0 : tensor<1x3xi32>
  }
}
