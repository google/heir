// RUN: heir-opt --polynomial-to-standard %s | FileCheck %s

#cycl_2048 = #_polynomial.polynomial<1 + x**1024>
#ring = #_polynomial.ring<cmod=4294967296, ideal=#cycl_2048>
#ring_prime = #_polynomial.ring<cmod=4294967291, ideal=#cycl_2048>
module {
  // CHECK-label: test_lower_from_tensor
  func.func @test_lower_from_tensor() -> !_polynomial.polynomial<#ring> {
    %c0 = arith.constant 0 : index
    // 2 + 2x + 5x^2
    %coeffs = arith.constant dense<[2, 2, 5]> : tensor<3xi32>
    // CHECK-NOT: _polynomial.from_tensor
    // CHECK: [[COEFFS:%.+]] = arith.constant dense<[2, 2, 5]> : tensor<3xi32>
    // CHECK: [[PAD:%.+]] = tensor.pad [[COEFFS]] low[0] high[1021]
    // CHECK: tensor<3xi32> to tensor<1024xi32>
    %poly = _polynomial.from_tensor %coeffs : tensor<3xi32> -> !_polynomial.polynomial<#ring>
    // CHECK: return
    return %poly : !_polynomial.polynomial<#ring>
  }

  // CHECK-label: f0
  // CHECK %arg0: tensor<1024xui64, #_polynomial.ring<cmod=4294967296, ideal=#_polynomial.polynomial<1 + x**1024>>>
  func.func @f0(%arg: !_polynomial.polynomial<#ring>) -> !_polynomial.polynomial<#ring> {
    return %arg : !_polynomial.polynomial<#ring>
  }

  // CHECK-label: test_lower_fn_and_call
  // CHECK-NOT: _polynomial.polynomial<#ring>
  func.func @test_lower_fn_and_call(%arg: !_polynomial.polynomial<#ring>) -> !_polynomial.polynomial<#ring>
 {
    %0 = func.call @f0(%arg) : (!_polynomial.polynomial<#ring>) -> !_polynomial.polynomial<#ring>
    return %0 : !_polynomial.polynomial<#ring>
  }

  func.func @test_i32_coeff_with_i32_mod() -> !_polynomial.polynomial<#ring_prime>
 {
    // CHECK: [[X:%.+]] = arith.constant dense<2> : [[TCOEFF:tensor<1024xi32>]]
    %coeffs1 = arith.constant dense<2> : tensor<1024xi32>
    // CHECK-NOT: _polynomial.from_tensor
    %poly0 = _polynomial.from_tensor %coeffs1 : tensor<1024xi32> -> !_polynomial.polynomial<#ring_prime>
    // CHECK: return
    return %poly0 : !_polynomial.polynomial<#ring_prime>

  }
}
