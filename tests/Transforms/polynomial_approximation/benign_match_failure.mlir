// RUN: heir-opt %s --polynomial-approximation 2> %t.err | FileCheck %s
// RUN: test ! -s %t.err
// RUN: heir-opt %s --polynomial-approximation=math-exp-method=taylor 2> %t.err | FileCheck %s
// RUN: test ! -s %t.err

// Cleartext operands cause notifyMatchFailure("operand is not secret").
// These operations must remain unchanged, with successful exit and no diagnostics.
// CHECK: func.func @cleartext
// CHECK: math.tanh
// CHECK: math.exp
// CHECK: arith.maximumf
// CHECK: return
func.func @cleartext(%x: f32) -> (f32, f32, f32) {
  %zero = arith.constant 0.0 : f32
  %tanh = math.tanh %x : f32
  %exp = math.exp %x : f32
  %relu = arith.maximumf %x, %zero : f32
  return %tanh, %exp, %relu : f32, f32, f32
}

// A secret binary operation with no constant operand is also a benign mismatch.
// CHECK: func.func @nonconstant_operands
// CHECK: arith.maximumf
// CHECK: return
func.func @nonconstant_operands(%x: f32 {secret.secret}, %y: f32 {secret.secret}) -> f32 {
  %max = arith.maximumf %x, %y : f32
  return %max : f32
}
