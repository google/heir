// RUN: heir-opt --polynomial-to-standard %s | FileCheck %s

#cycl_2048 = #polynomial.int_polynomial<1 + x**1024>
#ring = #polynomial.ring<coefficientType = i32, coefficientModulus = 4294967296 : i64, polynomialModulus=#cycl_2048>

// CHECK-label: test_lower_to_tensor
func.func @test_lower_to_tensor() -> tensor<1024xi32> {
  // CHECK: [[COEFFS:%.+]] = arith.constant
  %coeffs = arith.constant dense<2> : tensor<1024xi32>
  // CHECK-NOT: polynomial.from_tensor
  %poly = polynomial.from_tensor %coeffs : tensor<1024xi32> -> !polynomial.polynomial<ring=#ring>
  // CHECK-NOT: polynomial.to_tensor
  %tensor = polynomial.to_tensor %poly : !polynomial.polynomial<ring=#ring> -> tensor<1024xi32>
  // CHECK: return [[COEFFS]]
  return %tensor : tensor<1024xi32>
}

// CHECK-label: test_lower_to_tensor_small_coeffs
func.func @test_lower_to_tensor_small_coeffs() -> tensor<1024xi32> {
  // CHECK-NOT: polynomial.from_tensor
  // CHECK-NOT: polynomial.to_tensor
  // CHECK: [[COEFFS:%.+]] = arith.constant dense<[2, 2, 5]> : tensor<3xi32>
  // CHECK: [[PAD:%.+]] = tensor.pad [[COEFFS]] low[0] high[1021]
  // CHECK: tensor<3xi32> to tensor<1024xi32>
  %coeffs = arith.constant dense<[2, 2, 5]> : tensor<3xi32>
  %poly = polynomial.from_tensor %coeffs : tensor<3xi32> -> !polynomial.polynomial<ring=#ring>
  %tensor = polynomial.to_tensor %poly : !polynomial.polynomial<ring=#ring> -> tensor<1024xi32>
  // CHECK: return [[PAD]]
  return %tensor : tensor<1024xi32>
}
