// RUN: heir-opt -forward-insert-to-extract --split-input-file %s | FileCheck %s

module {
  func.func @main() -> i32 {
    // CHECK: %[[v0:.*]] = arith.constant -729 : i32
    // CHECK: return %[[v0]] : i32
    %c0_i32 = arith.constant 0 : i32
    %c610_i32 = arith.constant 610 : i32
    %c1954_i32 = arith.constant 1954 : i32
    %c-729_i32 = arith.constant -729 : i32
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %0 = tensor.empty() : tensor<1x3xi32>
    %inserted = tensor.insert %c0_i32 into %0[%c0, %c0] : tensor<1x3xi32>
    %inserted_0 = tensor.insert %c-729_i32 into %0[%c0, %c0] : tensor<1x3xi32>
    %inserted_1 = tensor.insert %c1954_i32 into %inserted_0[%c0, %c1] : tensor<1x3xi32>
    %inserted_2 = tensor.insert %c610_i32 into %inserted_1[%c0, %c2] : tensor<1x3xi32>
    %extracted = tensor.extract %inserted_2[%c0, %c0] : tensor<1x3xi32>
    func.return %extracted : i32
  }
}
