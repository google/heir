// RUN: heir-opt --polynomial-approximation %s | FileCheck %s

// max(a, b) and min(a, b) of two secrets become a +- max(., 0), whose ReLU
// shape is approximated.

// CHECK: @max_of_secrets
// CHECK: %[[diff:.*]] = arith.subf %[[b:.*]], %[[a:.*]] : tensor<4xf32>
// CHECK: %[[relu:.*]] = polynomial.eval {{.*}}%[[diff]]
// CHECK: arith.addf %[[a]], %[[relu]] : tensor<4xf32>
func.func @max_of_secrets(%a: tensor<4xf32> {secret.secret}, %b: tensor<4xf32> {secret.secret}) -> tensor<4xf32> {
  %0 = arith.maximumf %a, %b : tensor<4xf32>
  return %0 : tensor<4xf32>
}

// CHECK: @min_of_secrets
// CHECK: %[[diff:.*]] = arith.subf %[[a:.*]], %[[b:.*]] : f32
// CHECK: %[[relu:.*]] = polynomial.eval {{.*}}%[[diff]]
// CHECK: arith.subf %[[a]], %[[relu]] : f32
func.func @min_of_secrets(%a: f32 {secret.secret}, %b: f32 {secret.secret}) -> f32 {
  %0 = arith.minimumf %a, %b : f32
  return %0 : f32
}

// A max of cleartexts keeps its exact semantics.

// CHECK: @max_of_cleartexts
// CHECK: arith.maximumf
func.func @max_of_cleartexts(%a: tensor<4xf32>, %b: tensor<4xf32>) -> tensor<4xf32> {
  %0 = arith.maximumf %a, %b : tensor<4xf32>
  return %0 : tensor<4xf32>
}
