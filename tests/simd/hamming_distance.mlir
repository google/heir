// RUN: heir-opt --secretize=entry-function=hamming --wrap-generic --canonicalize --cse \
// RUN:   --full-loop-unroll --cse --canonicalize --insert-rotate --cse --canonicalize \
// RUN:   %s | FileCheck %s

// CHECK-LABEL: @hamming
// CHECK: secret.generic
// CHECK: arith.subi
// CHECK-NEXT: arith.muli
// CHECK-NEXT: tensor_ext.rotate
// CHECK-NEXT: arith.addi
// CHECK-NEXT: tensor_ext.rotate
// CHECK-NEXT: arith.addi
// CHECK-NEXT: tensor_ext.rotate
// CHECK-NEXT: arith.addi
// CHECK-NEXT: tensor.extract
// CHECK-NEXT: secret.yield

// TODO(#521): support rotate-and-reduce when the input is already a series of incremental rotations,
// as this IR is currently lowered to 4-1 rotate operations to sum after doing (x-y)**2 in SIMD.

func.func @hamming(%arg0: tensor<4xi16> {secret.secret}, %arg1: tensor<4xi16> {secret.secret}) -> i16 {
  %c0 = arith.constant 0 : index
  %c0_si16 = arith.constant 0 : i16
  %0 = affine.for %arg2 = 0 to 4 iter_args(%arg6 = %c0_si16) -> i16 {
    %1 = tensor.extract %arg0[%arg2] : tensor<4xi16>
    %2 = tensor.extract %arg1[%arg2] : tensor<4xi16>
    %3 = arith.subi %1, %2 : i16
    %4 = tensor.extract %arg0[%arg2] : tensor<4xi16>
    %5 = tensor.extract %arg1[%arg2] : tensor<4xi16>
    %6 = arith.subi %4, %5 : i16
    %7 = arith.muli %3, %6 : i16
    %8 = arith.addi %arg6, %7 : i16
    affine.yield %8 : i16
  }
  return %0 : i16
}
