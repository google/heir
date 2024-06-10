// RUN: heir-opt --polynomial-to-standard %s | FileCheck %s

#cycl_2048 = #polynomial.int_polynomial<1 + x**1024>
#ring = #polynomial.ring<coefficientType = i32, coefficientModulus = 4294967296 : i64, polynomialModulus=#cycl_2048>
#ring_prime = #polynomial.ring<coefficientType = i32, coefficientModulus = 4294967291 : i32, polynomialModulus=#cycl_2048>
module {
  // CHECK-label: test_lower_from_tensor
  func.func @test_lower_from_tensor() -> !polynomial.polynomial<ring=#ring> {
    %c0 = arith.constant 0 : index
    // 2 + 2x + 5x^2
    %coeffs = arith.constant dense<[2, 2, 5]> : tensor<3xi32>
    // CHECK-NOT: polynomial.from_tensor
    // CHECK: [[COEFFS:%.+]] = arith.constant dense<[2, 2, 5]> : tensor<3xi32>
    // CHECK: [[PAD:%.+]] = tensor.pad [[COEFFS]] low[0] high[1021]
    // CHECK: tensor<3xi32> to tensor<1024xi32>
    %poly = polynomial.from_tensor %coeffs : tensor<3xi32> -> !polynomial.polynomial<ring=#ring>
    // CHECK: return
    return %poly : !polynomial.polynomial<ring=#ring>
  }

  // CHECK-label: f0
  // CHECK %arg0: tensor<1024xui64, #polynomial.ring<coefficientType = i32, coefficientModulus = 4294967296 : i64, polynomialModulus=#polynomial.int_polynomial<1 + x**1024>>>
  func.func @f0(%arg: !polynomial.polynomial<ring=#ring>) -> !polynomial.polynomial<ring=#ring> {
    return %arg : !polynomial.polynomial<ring=#ring>
  }

  // CHECK-label: test_lower_fn_and_call
  // CHECK-NOT: polynomial.polynomial<#ring>
  func.func @test_lower_fn_and_call(%arg: !polynomial.polynomial<ring=#ring>) -> !polynomial.polynomial<ring=#ring>
 {
    %0 = func.call @f0(%arg) : (!polynomial.polynomial<ring=#ring>) -> !polynomial.polynomial<ring=#ring>
    return %0 : !polynomial.polynomial<ring=#ring>
  }

  func.func @test_i32_coeff_with_i32_mod() -> !polynomial.polynomial<ring=#ring_prime>
 {
    // CHECK: [[X:%.+]] = arith.constant dense<2> : [[TCOEFF:tensor<1024xi32>]]
    %coeffs1 = arith.constant dense<2> : tensor<1024xi32>
    // CHECK-NOT: polynomial.from_tensor
    %poly0 = polynomial.from_tensor %coeffs1 : tensor<1024xi32> -> !polynomial.polynomial<ring=#ring_prime>
    // CHECK: return
    return %poly0 : !polynomial.polynomial<ring=#ring_prime>

  }
}
