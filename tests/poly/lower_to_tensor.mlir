// RUN: heir-opt --poly-to-standard %s | FileCheck %s

#cycl_2048 = #poly.polynomial<1 + x**1024>
#ring = #poly.ring<cmod=4294967296, ideal=#cycl_2048>

// CHECK-label: test_lower_to_tensor
func.func @test_lower_to_tensor() -> tensor<1024xi32> {
  // CHECK: [[COEFFS:%.+]] = arith.constant
  %coeffs = arith.constant dense<2> : tensor<1024xi32>
  // CHECK-NOT: poly.from_tensor
  %poly = poly.from_tensor %coeffs : tensor<1024xi32> -> !poly.poly<#ring>
  // CHECK-NOT: poly.to_tensor
  %tensor = poly.to_tensor %poly : !poly.poly<#ring> -> tensor<1024xi32>
  // CHECK: return [[COEFFS]]
  return %tensor : tensor<1024xi32>
}

// CHECK-label: test_lower_to_tensor_small_coeffs
func.func @test_lower_to_tensor_small_coeffs() -> tensor<1024xi32> {
  // CHECK-NOT: poly.from_tensor
  // CHECK-NOT: poly.to_tensor
  // CHECK: [[COEFFS:%.+]] = arith.constant dense<[2, 2, 5]> : tensor<3xi32>
  // CHECK: [[PAD:%.+]] = tensor.pad [[COEFFS]] low[0] high[1021]
  // CHECK: tensor<3xi32> to tensor<1024xi32>
  %coeffs = arith.constant dense<[2, 2, 5]> : tensor<3xi32>
  %poly = poly.from_tensor %coeffs : tensor<3xi32> -> !poly.poly<#ring>
  %tensor = poly.to_tensor %poly : !poly.poly<#ring> -> tensor<1024xi32>
  // CHECK: return [[PAD]]
  return %tensor : tensor<1024xi32>
}
