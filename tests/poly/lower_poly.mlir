// RUN: heir-opt --lower-poly %s > %t
// RUN: FileCheck %s < %t

#cycl_2048 = #poly.polynomial<1 + x**1024>
#ring = #poly.ring<cmod=4294967296, ideal=#cycl_2048>
#ring_prime = #poly.ring<cmod=4294967291, ideal=#cycl_2048>
module {
  // CHECK-label: test_lower_from_tensor
  func.func @test_lower_from_tensor() {
    %c0 = arith.constant 0 : index
    // 2 + 2x + 5x^2
    %coeffs = arith.constant dense<[2, 2, 5]> : tensor<3xi32>
    // CHECK-NOT: poly.from_tensor
    // CHECK: [[COEFFS:%.+]] = arith.constant dense<[2, 2, 5]> : tensor<3xi32>
    // CHECK: [[EXT:%.+]] = arith.extui [[COEFFS]] : tensor<3xi32> to tensor<3xi64>
    // CHECK: [[PAD:%.+]] = tensor.pad [[EXT]] low[0] high[1021]
    // CHECK: tensor<3xi64> to tensor<1024xi64>
    %poly = poly.from_tensor %coeffs : tensor<3xi32> -> !poly.poly<#ring>
    // CHECK: return
    return
  }

  // CHECK-label: test_lower_to_tensor
  func.func @test_lower_to_tensor() -> tensor<1024xi64> {
    // CHECK: [[COEFFS:%.+]] = arith.constant
    %coeffs = arith.constant dense<2> : tensor<1024xi32>
    // CHECK-NOT: poly.from_tensor
    %poly = poly.from_tensor %coeffs : tensor<1024xi32> -> !poly.poly<#ring>
    // CHECK-NOT: poly.to_tensor
    // CHECK: [[EXT:%.+]] = arith.extui [[COEFFS]] :  tensor<1024xi32> to  tensor<1024xi64>
    %tensor = poly.to_tensor %poly : !poly.poly<#ring> -> tensor<1024xi64>
    // CHECK: return [[EXT]]
    return %tensor : tensor<1024xi64>
  }

  // CHECK-label: test_lower_to_tensor_small_coeffs
  func.func @test_lower_to_tensor_small_coeffs() -> tensor<1024xi64> {
    // CHECK-NOT: poly.from_tensor
    // CHECK-NOT: poly.to_tensor
    // CHECK: [[COEFFS:%.+]] = arith.constant dense<[2, 2, 5]> : tensor<3xi32>
    // CHECK: [[EXT:%.+]] = arith.extui [[COEFFS]] : tensor<3xi32> to tensor<3xi64>
    // CHECK: [[PAD:%.+]] = tensor.pad [[EXT]] low[0] high[1021]
    // CHECK: tensor<3xi64> to tensor<1024xi64>
    %coeffs = arith.constant dense<[2, 2, 5]> : tensor<3xi32>
    %poly = poly.from_tensor %coeffs : tensor<3xi32> -> !poly.poly<#ring>
    %tensor = poly.to_tensor %poly : !poly.poly<#ring> -> tensor<1024xi64>
    // CHECK: return [[PAD]]
    return %tensor : tensor<1024xi64>
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

  func.func @test_lower_add_power_of_two_cmod() -> !poly.poly<#ring> {
    // 2 + 2x + 2x^2 + ... + 2x^{1023}
    // CHECK: [[X:%.+]] = arith.constant dense<2> : [[T:tensor<1024xi32>]]
    %coeffs1 = arith.constant dense<2> : tensor<1024xi32>
    // CHECK: [[Y:%.+]] = arith.constant dense<3> : [[T]]
    %coeffs2 = arith.constant dense<3> : tensor<1024xi32>
    // CHECK-NOT: poly.from_tensor
    // CHECK: [[XEXT:%.+]] = arith.extui [[X]] : [[T]] to [[TPOLY:tensor<1024xi64>]]
    // CHECK: [[YEXT:%.+]] = arith.extui [[Y]] : [[T]] to [[TPOLY:tensor<1024xi64>]]
    %poly0 = poly.from_tensor %coeffs1 : tensor<1024xi32> -> !poly.poly<#ring>
    %poly1 = poly.from_tensor %coeffs2 : tensor<1024xi32> -> !poly.poly<#ring>
    // CHECK:  [[MOD:%.+]] = arith.constant dense<4294967296> : [[TPOLY]]
    // CHECK-NEXT: [[ADD:%.+]], [[OVERFLOW:%.+]] = arith.addui_extended [[XEXT]], [[YEXT]] : [[TPOLY]], tensor<1024xi1>
    // CHECK-NEXT: [[REM:%.+]] = arith.remui [[ADD]], [[MOD]] : [[TPOLY]]
    %poly2 = poly.add(%poly0, %poly1) {ring = #ring} : !poly.poly<#ring>
    // CHECK: return  [[REM]] : [[TPOLY]]
    return %poly2 : !poly.poly<#ring>
  }

  func.func @test_lower_add_prime_cmod() -> !poly.poly<#ring_prime> {
    // CHECK: [[X:%.+]] = arith.constant dense<2> : [[TCOEFF:tensor<1024xi31>]]
    %coeffs1 = arith.constant dense<2> : tensor<1024xi31>
    // CHECK: [[Y:%.+]] = arith.constant dense<3> : [[TCOEFF]]
    %coeffs2 = arith.constant dense<3> : tensor<1024xi31>
    // CHECK-NOT: poly.from_tensor
    // CHECK: [[XEXT:%.+]] = arith.extui [[X]] : [[TCOEFF]] to [[T:tensor<1024xi64>]]
    // CHECK: [[YEXT:%.+]] = arith.extui [[Y]] : [[TCOEFF]] to [[T:tensor<1024xi64>]]
    %poly0 = poly.from_tensor %coeffs1 : tensor<1024xi31> -> !poly.poly<#ring_prime>
    %poly1 = poly.from_tensor %coeffs2 : tensor<1024xi31> -> !poly.poly<#ring_prime>
    // CHECK:  [[MOD:%.+]] = arith.constant dense<4294967291> : [[T]]
    // CHECK-NEXT: [[ADD:%.+]], [[OVERFLOW:%.+]] = arith.addui_extended [[XEXT]], [[YEXT]] : [[T]], tensor<1024xi1>
    // CHECK-NEXT: [[REM:%.+]] = arith.remui [[ADD]], [[MOD]] : [[T]]
    // CHECK-NEXT: [[NMOD:%.+]] = arith.constant dense<25> : [[T]]
    // CHECK-NEXT: [[REMPLUS2N:%.+]] = arith.addi [[REM]], [[NMOD]] : [[T]]
    // CHECK-NEXT: [[RES:%.+]] = arith.select [[OVERFLOW]], [[REM]], [[REMPLUS2N]] : tensor<1024xi1>, [[T]]
    // CHECK-NEXT: [[RESMOD:%.+]] = arith.remui [[RES]], [[MOD]] : [[T]]
    %poly2 = poly.add(%poly0, %poly1) {ring = #ring_prime} : !poly.poly<#ring_prime>
    // CHECK: return  [[RESMOD]] : [[T]]
    return %poly2 : !poly.poly<#ring_prime>
  }

  func.func @test_i32_coeff_with_i32_mod() -> () {
    // CHECK: [[X:%.+]] = arith.constant dense<2> : [[TCOEFF:tensor<1024xi32>]]
    %coeffs1 = arith.constant dense<2> : tensor<1024xi32>
    // CHECK: [[Y:%.+]] = arith.constant dense<3> : [[TCOEFF]]
    %coeffs2 = arith.constant dense<3> : tensor<1024xi32>
    // CHECK-NOT: poly.from_tensor
    %poly0 = poly.from_tensor %coeffs1 : tensor<1024xi32> -> !poly.poly<#ring_prime>
    %poly1 = poly.from_tensor %coeffs2 : tensor<1024xi32> -> !poly.poly<#ring_prime>
    // CHECK: return
    return
  }
}
