// RUN: heir-opt --secretize=entry-function=hamming --wrap-generic \
// RUN:   --canonicalize --cse --heir-simd-vectorizer \
// RUN:   --secret-distribute-generic --secret-to-bgv \
// RUN:   %s | FileCheck %s

// CHECK-LABEL: @hamming
// CHECK: bgv.sub
// CHECK-NEXT: bgv.mul
// CHECK-NEXT: bgv.relinearize

// TODO(#521): After rotate-and-reduce works, only check for 10 bg.rotate
// CHECK-COUNT-1023: bgv.rotate
// CHECK: bgv.extract
// CHECK-NEXT: return

func.func @hamming(%arg0: tensor<1024xi16>, %arg1: tensor<1024xi16> {secret.secret}) -> i16 {
  %c0 = arith.constant 0 : index
  %c0_si16 = arith.constant 0 : i16
  %0 = affine.for %arg2 = 0 to 1024 iter_args(%arg6 = %c0_si16) -> i16 {
    %1 = tensor.extract %arg0[%arg2] : tensor<1024xi16>
    %2 = tensor.extract %arg1[%arg2] : tensor<1024xi16>
    %3 = arith.subi %1, %2 : i16
    %4 = tensor.extract %arg0[%arg2] : tensor<1024xi16>
    %5 = tensor.extract %arg1[%arg2] : tensor<1024xi16>
    %6 = arith.subi %4, %5 : i16
    %7 = arith.muli %3, %6 : i16
    %8 = arith.addi %arg6, %7 : i16
    affine.yield %8 : i16
  }
  return %0 : i16
}
