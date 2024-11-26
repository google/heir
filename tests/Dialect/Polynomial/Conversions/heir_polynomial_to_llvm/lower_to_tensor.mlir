// RUN: heir-opt --polynomial-to-mod-arith %s | FileCheck %s

#cycl_2048 = #polynomial.int_polynomial<1 + x**1024>
!coeff_ty = !mod_arith.int<65536:i32>
#ring = #polynomial.ring<coefficientType=!coeff_ty, polynomialModulus=#cycl_2048>

// CHECK-label: test_lower_to_tensor
func.func @test_lower_to_tensor() -> tensor<1024xi32> {
  // CHECK: [[COEFFS:%.+]] = arith.constant
  // CHECK: [[ENC:%.+]] = mod_arith.encapsulate [[COEFFS]]
  // CHECK: [[EXT:%.+]] = mod_arith.extract [[ENC]]
  %coeffsRaw = arith.constant dense<2> : tensor<1024xi32>
  %coeffs = mod_arith.encapsulate %coeffsRaw : tensor<1024xi32> -> tensor<1024x!coeff_ty>
  %poly = polynomial.from_tensor %coeffs : tensor<1024x!coeff_ty> -> !polynomial.polynomial<ring=#ring>
  %tensorMod = polynomial.to_tensor %poly : !polynomial.polynomial<ring=#ring> -> tensor<1024x!coeff_ty>
  %tensor = mod_arith.extract %tensorMod : tensor<1024x!coeff_ty> -> tensor<1024xi32>
  // CHECK: return [[EXT]]
  return %tensor : tensor<1024xi32>
}

// CHECK-label: test_lower_to_tensor_small_coeffs
func.func @test_lower_to_tensor_small_coeffs() -> tensor<1024xi32> {
  // CHECK-NOT: polynomial.from_tensor
  // CHECK-NOT: polynomial.to_tensor
  // CHECK: [[COEFFS:%.+]] = arith.constant dense<[2, 2, 5]> : tensor<3xi32>
  // CHECK: [[ENC:%.+]] = mod_arith.encapsulate [[COEFFS]]
  // CHECK: [[ZERO:%.+]] = mod_arith.constant 0
  // CHECK: [[PAD:%.+]] = tensor.pad [[ENC]] low[0] high[1021]
  // CHECK:   tensor.yield [[ZERO]]
  // CHECK: [[EXT:%.+]] = mod_arith.extract [[PAD]]
  // CHECK: return [[EXT]]
  %coeffsRaw = arith.constant dense<[2, 2, 5]> : tensor<3xi32>
  %coeffs = mod_arith.encapsulate %coeffsRaw : tensor<3xi32> -> tensor<3x!coeff_ty>
  %poly = polynomial.from_tensor %coeffs : tensor<3x!coeff_ty> -> !polynomial.polynomial<ring=#ring>
  %tensorMod = polynomial.to_tensor %poly : !polynomial.polynomial<ring=#ring> -> tensor<1024x!coeff_ty>
  %tensor = mod_arith.extract %tensorMod : tensor<1024x!coeff_ty> -> tensor<1024xi32>
  return %tensor : tensor<1024xi32>
}
