// RUN: heir-opt --compare-to-sign-rewrite --canonicalize %s | FileCheck %s

// On constants, each comparison becomes the sign of a folded difference, and
// it holds exactly when that difference is positive.

// CHECK: func.func @values
// CHECK-DAG: %[[NEG:.*]] = arith.constant -1.000000e+00 : f32
// CHECK-DAG: %[[POS:.*]] = arith.constant 1.000000e+00 : f32
// CHECK: math_ext.sign %[[POS]]
// CHECK: math_ext.sign %[[NEG]]
// CHECK: math_ext.sign %[[NEG]]
// CHECK: math_ext.sign %[[POS]]
// CHECK: math_ext.sign %[[POS]]
// CHECK: math_ext.sign %[[NEG]]
// CHECK: math_ext.sign %[[NEG]]
// CHECK: math_ext.sign %[[POS]]
func.func @values() -> (i1, i1, i1, i1, i1, i1, i1, i1) {
  %one = arith.constant 1.0 : f32
  %two = arith.constant 2.0 : f32
  %i1 = arith.constant 1 : i32
  %i2 = arith.constant 2 : i32
  %0 = arith.cmpf olt, %one, %two : f32
  %1 = arith.cmpf olt, %two, %one : f32
  %2 = arith.cmpf ogt, %one, %two : f32
  %3 = arith.cmpf ogt, %two, %one : f32
  %4 = arith.cmpi slt, %i1, %i2 : i32
  %5 = arith.cmpi slt, %i2, %i1 : i32
  %6 = arith.cmpi sgt, %i1, %i2 : i32
  %7 = arith.cmpi sgt, %i2, %i1 : i32
  return %0, %1, %2, %3, %4, %5, %6, %7 : i1, i1, i1, i1, i1, i1, i1, i1
}
