// RUN: heir-opt --lower-poly %s > %t
// RUN: FileCheck %s < %t

#cycl_2048 = #poly.polynomial<1 + x**1024>
#ring = #poly.ring<cmod=4294967296, ideal=#cycl_2048>
module {
  func.func @test_lower_from_tensor() {
    %c0 = arith.constant 0 : index
    // 2 + 2x + 5x^2
    %coeffs = arith.constant dense<[2, 2, 5]> : tensor<3xi32>
    // CHECK-NOT: poly.from_tensor
    %poly = poly.from_tensor %coeffs : tensor<3xi32> -> !poly.poly<#ring>
    return
  }

  // CHECK-label: f0
  // CHECK %arg0: tensor<1024xui64, #poly.ring<cmod=4294967296, ideal=#poly.polynomial<1 + x**1024>>>
  func.func @f0(%arg: !poly.poly<#ring>) -> !poly.poly<#ring> {
    return %arg : !poly.poly<#ring>
  }

  // CHECK-label: test_lower_fn_and_call
  // CHECK-NOT: poly.poly<#ring>
  func.func @test_lower_fn_and_call(%arg: !poly.poly<#ring>) {
    func.call @f0(%arg) : (!poly.poly<#ring>) -> !poly.poly<#ring>
    return
  }
}
