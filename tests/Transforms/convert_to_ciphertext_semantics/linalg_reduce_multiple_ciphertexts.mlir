// RUN: heir-opt %s --layout-propagation=min-slot-count=16 --convert-to-ciphertext-semantics=min-slot-count=16 --verify-diagnostics

// Rotations act within one ciphertext, so the rotate-and-reduce kernel
// rejects an input packed into two. The second error comes from the verifier
// after the conversion retypes the operands of the rejected op.

func.func @row_sum(%x: !secret.secret<tensor<4x8xf32>>) -> !secret.secret<tensor<4xf32>> {
  %zero = arith.constant dense<0.0> : tensor<4xf32>
  %0 = secret.generic(%x: !secret.secret<tensor<4x8xf32>>) {
  ^body(%input: tensor<4x8xf32>):
    // expected-error@+2 {{linalg.reduce only supported with an input in one ciphertext}}
    // expected-error@+1 {{'linalg.reduce' op number of dimensions after reduction 1 doesn't match the init rank 2}}
    %sum = linalg.reduce { arith.addf } ins(%input : tensor<4x8xf32>) outs(%zero : tensor<4xf32>) dimensions = [1]
    secret.yield %sum : tensor<4xf32>
  } -> !secret.secret<tensor<4xf32>>
  return %0 : !secret.secret<tensor<4xf32>>
}
