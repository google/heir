// RUN: heir-opt --activation-canonicalizations --canonicalize %s | FileCheck %s

// CHECK: func.func @relu_with_domain_attributes
// CHECK-SAME: (%[[ARG0:.*]]: f32) -> f32
// CHECK: %[[A:.*]] = arith.constant 0
// CHECK-NEXT: arith.maximumf %[[ARG0]], %[[A]] {domain_lower = -6.670000e+00 : f64, domain_upper = 6.550000e+00 : f64}
// CHECK: return
func.func @relu_with_domain_attributes(%arg0: f32) -> f32 {
  %cst_0 = arith.constant 0.000000e+00 : f32
  %0 = arith.cmpf ugt, %arg0, %cst_0 : f32
  %1 = arith.select %0, %arg0, %cst_0 {domain_lower = -6.67 : f64, domain_upper = 6.55 : f64} : f32
  return %1 : f32
}
