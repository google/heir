// Ported from https://github.com/MarbleHE/HECO/blob/3e13744233ab0c09030a41ef98b4e061b6fa2eac/evaluation/benchmark/heco_input/hammingdistance_4096.mlir

// RUN: heir-opt --secretize --wrap-generic --canonicalize --cse --heco-simd-vectorizer %s | FileCheck %s

// CHECK: @hamming
// CHECK-NOT: affine.for
// CHECK: arith.subi
// CHECK: arith.muli
// CHECK-COUNT-12: tensor_ext.rotate
// CHECK-NOT: tensor_ext.rotate
// CHECK: tensor.extract

func.func @hamming(%arg0: tensor<4096xi16>, %arg1: tensor<4096xi16>) -> i16 {
  %c0_si16 = arith.constant 0 : i16
  %0 = affine.for %arg2 = 0 to 4096 iter_args(%arg6 = %c0_si16) -> i16 {
    %1 = tensor.extract %arg0[%arg2] : tensor<4096xi16>
    %2 = tensor.extract %arg1[%arg2] : tensor<4096xi16>
    %3 = arith.subi %1, %2 : i16
    %4 = tensor.extract %arg0[%arg2] : tensor<4096xi16>
    %5 = tensor.extract %arg1[%arg2] : tensor<4096xi16>
    %6 = arith.subi %4, %5 : i16
    %7 = arith.muli %3, %6 : i16
    %8 = arith.addi %arg6, %7 : i16
    affine.yield %8 : i16
  }
  return %0 : i16
}
