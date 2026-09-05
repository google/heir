// RUN: heir-opt --softmax-to-ns-softmax --split-input-file %s | FileCheck %s

// Normalize-and-square: shift by M = domain_upper, /2^k compression (k=2
// default), exp with the compressed domain [(lo-M)/2^k, 0], then k rounds
// of square -> last-dim reduce -> recip with STRUCTURAL domains (row
// length n=32: intermediate [1/128, 64], final [1/64, 1.25]) ->
// broadcast -> multiply.

// CHECK: func.func @softmax_ns_stamped
// CHECK: arith.constant dense<4.500000e+00>
// CHECK: arith.subf
// CHECK: arith.constant dense<2.500000e-01>
// CHECK: arith.mulf
// CHECK: math.exp
// CHECK-SAME: degree = 13
// CHECK-SAME: domain_lower = -2.875000e+00
// CHECK-SAME: domain_upper = 0.000000e+00
// CHECK: arith.mulf
// CHECK: linalg.reduce
// CHECK: math_ext.recip
// CHECK-SAME: domain_lower = 7.812500e-03
// CHECK-SAME: domain_upper = 6.400000e+01
// CHECK: linalg.broadcast
// CHECK: arith.mulf
// CHECK: arith.mulf
// CHECK: linalg.reduce
// CHECK: math_ext.recip
// CHECK-SAME: domain_lower = 1.562500e-02
// CHECK-SAME: domain_upper = 1.250000e+00
// CHECK: linalg.broadcast
// CHECK: arith.mulf
// CHECK-NOT: math_ext.softmax
func.func @softmax_ns_stamped(%arg0: tensor<2x32x32xf32>) -> tensor<2x32x32xf32> {
  %1 = math_ext.softmax %arg0 {domain_lower = -7.0 : f64,
                               domain_upper = 4.5 : f64,
                               exp_degree = 13 : i32}
      : tensor<2x32x32xf32>
  return %1 : tensor<2x32x32xf32>
}

// -----

// Defaults: M = 6.0, exp deg 9, exp domain (-9 - 6)/4 = -3.75.

// CHECK: func.func @softmax_ns_defaults
// CHECK: arith.constant dense<6.000000e+00>
// CHECK: math.exp
// CHECK-SAME: degree = 9
// CHECK-SAME: domain_lower = -3.750000e+00
// CHECK-NOT: math_ext.softmax
func.func @softmax_ns_defaults(%arg0: tensor<32x32xf32>) -> tensor<32x32xf32> {
  %1 = math_ext.softmax %arg0 : tensor<32x32xf32>
  return %1 : tensor<32x32xf32>
}
