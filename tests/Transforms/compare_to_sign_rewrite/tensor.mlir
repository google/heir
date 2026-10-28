// RUN: heir-opt --compare-to-sign-rewrite %s | FileCheck %s

// Comparisons of tensors (e.g. from the generic linalg kernel) use splat
// constants of the operand shape.

// CHECK: func.func @cmpf_tensor
// CHECK: math_ext.sign
// CHECK-DAG: arith.constant dense<1.000000e+00> : tensor<1x8xf32>
// CHECK-DAG: arith.constant dense<5.000000e-01> : tensor<1x8xf32>
// CHECK: arith.fptosi %{{.*}} : tensor<1x8xf32> to tensor<1x8xi1>
func.func @cmpf_tensor(%a: tensor<1x8xf32>, %b: tensor<1x8xf32>) -> tensor<1x8xi1> {
  %0 = arith.cmpf ogt, %a, %b : tensor<1x8xf32>
  return %0 : tensor<1x8xi1>
}

// CHECK: func.func @cmpi_tensor
// CHECK: arith.index_cast %{{.*}} : tensor<1x8xindex> to tensor<1x8xi32>
// CHECK: arith.sitofp %{{.*}} : tensor<1x8xi32> to tensor<1x8xf32>
// CHECK: math_ext.sign
func.func @cmpi_tensor(%a: tensor<1x8xindex>, %b: tensor<1x8xindex>) -> tensor<1x8xi1> {
  %0 = arith.cmpi ugt, %a, %b : tensor<1x8xindex>
  return %0 : tensor<1x8xi1>
}

// CHECK: func.func @cmpi_ugt
// CHECK: %[[lhs:.*]] = arith.sitofp %arg0
// CHECK: %[[rhs:.*]] = arith.sitofp %arg1
// CHECK: arith.subf %[[lhs]], %[[rhs]] : f32
// CHECK: math_ext.sign
func.func @cmpi_ugt(%a: i32, %b: i32) -> i1 {
  %0 = arith.cmpi ugt, %a, %b : i32
  return %0 : i1
}
