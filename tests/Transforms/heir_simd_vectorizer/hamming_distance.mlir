// RUN: heir-opt --secretize --wrap-generic --canonicalize --cse \
// RUN:   --heir-simd-vectorizer %s | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-ARITH

// RUN: heir-opt --arith-to-mod-arith --secretize --wrap-generic --canonicalize --cse \
// RUN:   --heir-simd-vectorizer %s | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-MOD-ARITH

// CHECK-LABEL: @hamming
// CHECK: secret.generic
// CHECK-ARITH: arith.subi
// CHECK-MOD-ARITH: mod_arith.sub
// CHECK-ARITH-NEXT: arith.muli
// CHECK-MOD-ARITH-NEXT: mod_arith.mul
// CHECK-NEXT: tensor_ext.rotate
// CHECK-ARITH-NEXT: arith.addi
// CHECK-MOD-ARITH-NEXT: mod_arith.add
// CHECK-NEXT: tensor_ext.rotate
// CHECK-ARITH-NEXT: arith.addi
// CHECK-MOD-ARITH-NEXT: mod_arith.add
// CHECK-NEXT: tensor.extract
// CHECK-NEXT: secret.yield

func.func @hamming(%arg0: tensor<4xi16>, %arg1: tensor<4xi16>) -> i16 {
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
