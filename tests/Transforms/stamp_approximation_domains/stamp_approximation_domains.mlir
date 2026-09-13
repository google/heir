// RUN: heir-opt --stamp-approximation-domains %s | FileCheck %s

// CHECK: @stamp_bare_rsqrt
func.func @stamp_bare_rsqrt(%x: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: math.rsqrt
  // CHECK-SAME: degree = 12
  // CHECK-SAME: domain_lower = 5.000000e-01
  // CHECK-SAME: domain_upper = 5.000000e+00
  %0 = math.rsqrt %x : tensor<4xf32>
  return %0 : tensor<4xf32>
}

// CHECK: @stamp_bare_erf
func.func @stamp_bare_erf(%x: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: math.erf
  // CHECK-SAME: degree = 27
  // CHECK-SAME: domain_lower = -9.000000e+00
  // CHECK-SAME: domain_upper = 9.000000e+00
  %0 = math.erf %x : tensor<4xf32>
  return %0 : tensor<4xf32>
}

// CHECK: @respect_existing_stamps
func.func @respect_existing_stamps(%x: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: math.rsqrt
  // CHECK-SAME: domain_lower = 1.000000e-01
  // CHECK-NOT: degree
  %0 = math.rsqrt %x {domain_lower = 0.1 : f64, domain_upper = 2.0 : f64} : tensor<4xf32>
  return %0 : tensor<4xf32>
}
