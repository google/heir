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
}
