// RUN: heir-opt --polynomial-approximation --canonicalize %s | FileCheck %s

// A secret divided by a plaintext becomes a multiplication by the cleartext
// reciprocal, which folds away for a constant divisor (e.g., torch's GELU
// divides by sqrt(2)).

// CHECK: @divide_by_constant
// CHECK-SAME: (%[[X:.*]]: tensor<4xf32>
// CHECK: %[[RECIP:.*]] = arith.constant dense<0.707106{{[0-9]*}}> : tensor<4xf32>
// CHECK: %[[RESULT:.*]] = arith.mulf %[[X]], %[[RECIP]] : tensor<4xf32>
// CHECK-NOT: arith.divf
// CHECK: return %[[RESULT]]
func.func @divide_by_constant(%x: tensor<4xf32> {secret.secret}) -> tensor<4xf32> {
  %sqrt2 = arith.constant dense<1.41421354> : tensor<4xf32>
  %0 = arith.divf %x, %sqrt2 : tensor<4xf32>
  return %0 : tensor<4xf32>
}

// A non-constant plaintext divisor keeps a cleartext reciprocal.
// CHECK: @divide_by_plaintext
// CHECK-SAME: (%[[X:.*]]: f32 {secret.secret}, %[[P:.*]]: f32)
// CHECK: %[[ONE:.*]] = arith.constant 1.000000e+00 : f32
// CHECK: %[[RECIP:.*]] = arith.divf %[[ONE]], %[[P]] : f32
// CHECK: %[[RESULT:.*]] = arith.mulf %[[X]], %[[RECIP]] : f32
// CHECK: return %[[RESULT]]
func.func @divide_by_plaintext(%x: f32 {secret.secret}, %p: f32) -> f32 {
  %0 = arith.divf %x, %p : f32
  return %0 : f32
}

// A division of two plaintexts is left alone.
// CHECK: @divide_plaintexts
// CHECK: %[[RESULT:.*]] = arith.divf
// CHECK-NOT: arith.mulf
// CHECK: return %[[RESULT]]
func.func @divide_plaintexts(%a: f32, %b: f32) -> f32 {
  %0 = arith.divf %a, %b : f32
  return %0 : f32
}
