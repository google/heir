// RUN: heir-opt --poly-to-standard %s | FileCheck %s

#cycl_2048 = #poly.polynomial<1 + x**1024>
#ring = #poly.ring<cmod=4294967296, ideal=#cycl_2048>
#ring_prime = #poly.ring<cmod=4294967291, ideal=#cycl_2048>
module {
  // CHECK-label: test_lower_from_tensor
  func.func @test_lower_from_tensor() -> !poly.poly<#ring> {
    %c0 = arith.constant 0 : index
    // 2 + 2x + 5x^2
    %coeffs = arith.constant dense<[2, 2, 5]> : tensor<3xi32>
    // CHECK-NOT: poly.from_tensor
    // CHECK: [[COEFFS:%.+]] = arith.constant dense<[2, 2, 5]> : tensor<3xi32>
    // CHECK: [[PAD:%.+]] = tensor.pad [[COEFFS]] low[0] high[1021]
    // CHECK: tensor<3xi32> to tensor<1024xi32>
    %poly = poly.from_tensor %coeffs : tensor<3xi32> -> !poly.poly<#ring>
    // CHECK: return
    return %poly : !poly.poly<#ring>
  }

  // CHECK-label: f0
  // CHECK %arg0: tensor<1024xui64, #poly.ring<cmod=4294967296, ideal=#poly.polynomial<1 + x**1024>>>
  func.func @f0(%arg: !poly.poly<#ring>) -> !poly.poly<#ring> {
    return %arg : !poly.poly<#ring>
  }

  // CHECK-label: test_lower_fn_and_call
  // CHECK-NOT: poly.poly<#ring>
  func.func @test_lower_fn_and_call(%arg: !poly.poly<#ring>) -> !poly.poly<#ring>
 {
    %0 = func.call @f0(%arg) : (!poly.poly<#ring>) -> !poly.poly<#ring>
    return %0 : !poly.poly<#ring>
  }

  func.func @test_i32_coeff_with_i32_mod() -> !poly.poly<#ring_prime>
 {
    // CHECK: [[X:%.+]] = arith.constant dense<2> : [[TCOEFF:tensor<1024xi32>]]
    %coeffs1 = arith.constant dense<2> : tensor<1024xi32>
    // CHECK-NOT: poly.from_tensor
    %poly0 = poly.from_tensor %coeffs1 : tensor<1024xi32> -> !poly.poly<#ring_prime>
    // CHECK: return
    return %poly0 : !poly.poly<#ring_prime>

  }
}
