// RUN: heir-opt --softmax-to-ns-softmax %s | FileCheck %s

// Padded scores: the NS lowering masks ONCE
// after exp with the logical-region indicator, pins pad-row denominators
// (+1 on rows >= 4), and uses the LOGICAL row length n=4 in the
// structural recip domains: intermediate [1/16, 8], final [1/8, 1.25].

// CHECK: @padded_softmax
// Exp domain: (domain_lower - M)/2^k = (-6 - 3)/4 = -2.25.
// CHECK: math.exp
// CHECK-SAME: domain_lower = -2.250000e+00
// After exp: pads hold the compile-time constant P(-M/2^k) = P(-0.75)
// of the degree-9 CF polynomial on [-2.25, 0]; one depth-free plaintext
// ADD of -P(-0.75) at the pad positions restores exact zeros (replacing
// the mask multiply).
// CHECK: arith.constant dense<{{\[\[}}0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, -0.472366542]
// CHECK: arith.addf
// Pad-row pin constant (1 on rows >= logical 4), then round 1: square,
// reduce, pinned sum, recip on the intermediate structural domain.
// CHECK: arith.constant dense<[0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 1.000000e+00]>
// CHECK: arith.mulf
// CHECK: linalg.reduce
// CHECK: arith.addf
// CHECK: math_ext.recip
// CHECK-SAME: domain_lower = 6.250000e-02
// CHECK-SAME: domain_upper = 8.000000e+00
// Round 2: final structural domain from the LOGICAL n = 4.
// CHECK: arith.addf
// CHECK: math_ext.recip
// CHECK-SAME: domain_lower = 1.250000e-01
// CHECK-SAME: domain_upper = 1.250000e+00
// CHECK-NOT: math_ext.softmax
func.func @padded_softmax(%q: tensor<4x4xf32>, %k: tensor<4x4xf32>) -> tensor<5x5xf32> {
  %cst = arith.constant 0.0 : f32
  %q_pad = tensor.pad %q low[0, 0] high[1, 1] {
  ^bb0(%i: index, %j: index):
    tensor.yield %cst : f32
  } : tensor<4x4xf32> to tensor<5x5xf32>
  %k_pad = tensor.pad %k low[0, 0] high[1, 1] {
  ^bb0(%i: index, %j: index):
    tensor.yield %cst : f32
  } : tensor<4x4xf32> to tensor<5x5xf32>
  %init = arith.constant dense<0.0> : tensor<5x5xf32>
  %scores = linalg.matmul {tensor_ext.padding = #tensor_ext.padding<logicalShape = [4, 4], paddedShape = [5, 5], zeroPadded = true>}
      ins(%q_pad, %k_pad : tensor<5x5xf32>, tensor<5x5xf32>)
      outs(%init : tensor<5x5xf32>) -> tensor<5x5xf32>
  %sm = math_ext.softmax %scores {domain_lower = -6.0 : f64, domain_upper = 3.0 : f64} : tensor<5x5xf32>
  return %sm : tensor<5x5xf32>
}
