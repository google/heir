// RUN: heir-opt --lower-recip --split-input-file %s | FileCheck %s

// Affine seed (1 mul + 1 add) followed by 6 Goldschmidt iterations
// (2 muls + 1 sub each): 13 mulf total, no division anywhere.
// CHECK: func.func @recip_tensor
// CHECK-COUNT-13: arith.mulf
// CHECK-NOT: math_ext.recip
// CHECK-NOT: arith.divf
func.func @recip_tensor(%arg0: tensor<2x32xf32> {secret.secret})
    -> tensor<2x32xf32> {
  %0 = math_ext.recip %arg0 {domain_lower = 1.000000e-01 : f64,
                             domain_upper = 3.500000e+00 : f64}
      : tensor<2x32xf32>
  return %0 : tensor<2x32xf32>
}

// -----

// A WIDE (conservatively stamped) domain automatically gets more
// iterations: [0.02, 40] has seed error E = 0.996, needing 12
// iterations for 1e-4 (1 + 2*12 = 25 mulf) — the num-iterations
// option is only a floor.
// CHECK: func.func @recip_wide_domain
// CHECK-COUNT-25: arith.mulf
// CHECK-NOT: math_ext.recip
func.func @recip_wide_domain(%arg0: tensor<2x32xf32> {secret.secret})
    -> tensor<2x32xf32> {
  %0 = math_ext.recip %arg0 {domain_lower = 2.000000e-02 : f64,
                             domain_upper = 4.000000e+01 : f64}
      : tensor<2x32xf32>
  return %0 : tensor<2x32xf32>
}

// -----

// A tight domain derives fewer than the floor and stays at exactly the
// historical 6 iterations (13 mulf): existing circuits are unchanged.
// CHECK: func.func @recip_floor_domain
// CHECK-COUNT-13: arith.mulf
// CHECK-NOT: math_ext.recip
func.func @recip_floor_domain(%arg0: tensor<2x32xf32> {secret.secret})
    -> tensor<2x32xf32> {
  %0 = math_ext.recip %arg0 {domain_lower = 9.000000e-01 : f64,
                             domain_upper = 2.050000e+01 : f64}
      : tensor<2x32xf32>
  return %0 : tensor<2x32xf32>
}

// -----

// Without domain attributes there is no sound lowering: the op is left
// alone (and fails loudly downstream).
// CHECK: func.func @recip_no_domain
// CHECK: math_ext.recip
func.func @recip_no_domain(%arg0: tensor<2x32xf32> {secret.secret})
    -> tensor<2x32xf32> {
  %0 = math_ext.recip %arg0 : tensor<2x32xf32>
  return %0 : tensor<2x32xf32>
}
