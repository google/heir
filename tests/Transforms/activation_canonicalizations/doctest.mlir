// RUN: heir-opt --activation-canonicalizations --canonicalize %s

// CHECK: func.func @relu
// CHECK-SAME: (%[[ARG0:.*]]: f32) -> f32
// CHECK: %[[A:.*]] = arith.constant 0
// CHECK-NEXT: arith.maximumf %[[ARG0]], %[[A]]
// CHECK: return
func.func @relu(%arg0: f32) -> f32 {
  %cst_0 = arith.constant 0.000000e+00 : f32
  %0 = arith.cmpf ugt, %arg0, %cst_0 : f32
  %1 = arith.select %0, %arg0, %cst_0 : f32
  return %1 : f32
}
