// RUN: heir-opt --layout-propagation=ciphertext-size=1024 %s | FileCheck %s

// CHECK: @dot_product
func.func @dot_product(
      %arg0: !secret.secret<tensor<8xi16>>,
      %arg1: !secret.secret<tensor<8xi16>>) -> !secret.secret<i16> {
  %c0_i16 = arith.constant 0 : i16
  %0 = secret.generic ins(%arg0, %arg1 : !secret.secret<tensor<8xi16>>, !secret.secret<tensor<8xi16>>) {
  ^body(%input0: tensor<8xi16>, %input1: tensor<8xi16>):
    %1 = affine.for %arg2 = 0 to 8 iter_args(%arg3 = %c0_i16) -> (i16) {
      %extracted = tensor.extract %input0[%arg2] : tensor<8xi16>
      %extracted_0 = tensor.extract %input1[%arg2] : tensor<8xi16>
      %2 = arith.muli %extracted, %extracted_0 : i16
      %3 = arith.addi %arg3, %2 : i16
      affine.yield %3 : i16
    }
    secret.yield %1 : i16
  } -> !secret.secret<i16>
  return %0 : !secret.secret<i16>
}
