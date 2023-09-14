// RUN: heir-opt --lower-poly %s > %t
// RUN: FileCheck %s < %t

#cycl_2048 = #poly.polynomial<1 + x**1024>
#ring = #poly.ring<cmod=4294967296, ideal=#cycl_2048>
module {
  // CHECK-label: f0
  // CHECK %arg0: tensor<2x1024xui64, #poly.ring<cmod=4294967296, ideal=#poly.polynomial<1 + x**1024>>>
  func.func @f0(%arg: tensor<2x!poly.poly<#ring>>) -> tensor<2x!poly.poly<#ring>> {
    return %arg : tensor<2x!poly.poly<#ring>>
  }

  // CHECK-label: test_lower_fn_and_call
  // CHECK-NOT: poly.poly<#ring>
  func.func @test_lower_fn_and_call(%arg: tensor<2x!poly.poly<#ring>>) {
    func.call @f0(%arg) : (tensor<2x!poly.poly<#ring>>) -> tensor<2x!poly.poly<#ring>>
    return
  }

  // CHECK-label: test_lower_tensor_poly_to_tensor
  func.func @test_lower_tensor_poly_to_tensor() -> (tensor<2x!poly.poly<#ring>>) {
    // CHECK: [[COEFFS:%.+]] = arith.constant
    %coeffs = arith.constant dense<2> : tensor<1024xi32>
    // CHECK-NOT: poly.from_tensor
    // CHECK: [[EXT:%.+]] = arith.extui [[COEFFS]] :  tensor<1024xi32> to  tensor<1024xi64>
    %poly = poly.from_tensor %coeffs : tensor<1024xi32> -> !poly.poly<#ring>
    // CHECK: [[TENSOR:%.+]] = tensor.from_elements [[EXT]], [[EXT]] : tensor<2x1024xi64>
    %tensor_poly = tensor.from_elements %poly, %poly : tensor<2x!poly.poly<#ring>>
    // CHECK: return [[TENSOR]] : tensor<2x1024xi64>
    return %tensor_poly : tensor<2x!poly.poly<#ring>>
  }
}
