// RUN: heir-opt --split-input-file --activation-canonicalizations %s | FileCheck %s

#map = affine_map<(d0, d1) -> (d0, d1)>

// A torch ReLU imports as a linalg.generic carrying the polynomial-approximation
// domain on the generic op, with cmpf+select in its body. The float select
// must canonicalize to arith.maximumf AND inherit the generic's
// domain_lower/domain_upper so PolynomialApproximation reads the right domain.

// The domain must move onto the maximumf and NOT remain on the generic
// (leaving it on both collides during later activation lifting).
// CHECK: func.func @relu_generic
// CHECK: linalg.generic
// CHECK-NOT: domain_lower
// CHECK: %[[MAX:.*]] = arith.maximumf
// CHECK-SAME: domain_lower = -0.78033679723739624
// CHECK-SAME: domain_upper = 0.27823492884635925
// CHECK: linalg.yield %[[MAX]]
// CHECK: return
func.func @relu_generic(%arg0: tensor<1x3xf32>) -> tensor<1x3xf32> {
  %0 = tensor.empty() : tensor<1x3xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"], domain_lower = -0.78033679723739624 : f64, domain_upper = 0.27823492884635925 : f64} ins(%arg0 : tensor<1x3xf32>) outs(%0 : tensor<1x3xf32>) {
  ^bb0(%in: f32, %out: f32):
    %2 = arith.cmpf ugt, %in, %cst : f32
    %3 = arith.select %2, %in, %cst : f32
    linalg.yield %3 : f32
  } -> tensor<1x3xf32>
  return %1 : tensor<1x3xf32>
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>

// A fused elementwise generic can carry more than one ReLU. Every maximumf must
// get the domain: the generic's bounds are copied per-ReLU and only stripped
// once all of them have their own copy. Forwarding onto the first select and
// stripping there would leave the second on the default [-1, 1] domain.
// CHECK: func.func @two_relus_in_one_generic
// CHECK: linalg.generic
// CHECK-NOT: domain_lower
// CHECK: %[[MAX0:.*]] = arith.maximumf
// CHECK-SAME: domain_lower = -5.000000e+00
// CHECK-SAME: domain_upper = 5.000000e+00
// CHECK: %[[MAX1:.*]] = arith.maximumf
// CHECK-SAME: domain_lower = -5.000000e+00
// CHECK-SAME: domain_upper = 5.000000e+00
// CHECK: arith.addf %[[MAX0]], %[[MAX1]]
// CHECK: return
func.func @two_relus_in_one_generic(%arg0: tensor<1x3xf32>, %arg1: tensor<1x3xf32>) -> tensor<1x3xf32> {
  %0 = tensor.empty() : tensor<1x3xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %1 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"], domain_lower = -5.0 : f64, domain_upper = 5.0 : f64} ins(%arg0, %arg1 : tensor<1x3xf32>, tensor<1x3xf32>) outs(%0 : tensor<1x3xf32>) {
  ^bb0(%in: f32, %in2: f32, %out: f32):
    %2 = arith.cmpf ugt, %in, %cst : f32
    %3 = arith.select %2, %in, %cst : f32
    %4 = arith.cmpf ugt, %in2, %cst : f32
    %5 = arith.select %4, %in2, %cst : f32
    %6 = arith.addf %3, %5 : f32
    linalg.yield %6 : f32
  } -> tensor<1x3xf32>
  return %1 : tensor<1x3xf32>
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>

// Bounds on BOTH the select and the enclosing generic: the select's win (they
// are the more specific annotation), and the generic's redundant copy is still
// dropped so the domain ends up on exactly one op.
// CHECK: func.func @domain_on_select_and_generic
// CHECK: linalg.generic
// CHECK-NOT: domain_lower
// CHECK: arith.maximumf
// CHECK-SAME: domain_lower = -1.000000e+00
// CHECK-SAME: domain_upper = 1.000000e+00
// CHECK: return
func.func @domain_on_select_and_generic(%arg0: tensor<1x3xf32>) -> tensor<1x3xf32> {
  %0 = tensor.empty() : tensor<1x3xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"], domain_lower = -5.0 : f64, domain_upper = 5.0 : f64} ins(%arg0 : tensor<1x3xf32>) outs(%0 : tensor<1x3xf32>) {
  ^bb0(%in: f32, %out: f32):
    %2 = arith.cmpf ugt, %in, %cst : f32
    %3 = arith.select %2, %in, %cst {domain_lower = -1.0 : f64, domain_upper = 1.0 : f64} : f32
    linalg.yield %3 : f32
  } -> tensor<1x3xf32>
  return %1 : tensor<1x3xf32>
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>

// No ReLU in the body, so there is nothing to forward to: the generic keeps its
// bounds rather than having them silently discarded.
// CHECK: func.func @no_relu_keeps_generic_domain
// CHECK: linalg.generic
// CHECK-SAME: domain_lower = -5.000000e+00
// CHECK-SAME: domain_upper = 5.000000e+00
// CHECK: return
func.func @no_relu_keeps_generic_domain(%arg0: tensor<1x3xf32>) -> tensor<1x3xf32> {
  %0 = tensor.empty() : tensor<1x3xf32>
  %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"], domain_lower = -5.0 : f64, domain_upper = 5.0 : f64} ins(%arg0 : tensor<1x3xf32>) outs(%0 : tensor<1x3xf32>) {
  ^bb0(%in: f32, %out: f32):
    %2 = arith.mulf %in, %in : f32
    linalg.yield %2 : f32
  } -> tensor<1x3xf32>
  return %1 : tensor<1x3xf32>
}
