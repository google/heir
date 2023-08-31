// RUN: heir-opt --lower-poly %s > %t
// RUN: FileCheck %s < %t

#cycl_2048 = #poly.polynomial<1 + x**1024>
#ring = #poly.ring<cmod=4294967296, ideal=#cycl_2048>
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

  func.func @test_lower_from_coeffs() -> !poly.poly<#ring> {
    // 2 + 2x + 2x^2 + ... + 2x^{1023}
    // CHECK: [[X:%.+]] = arith.constant dense<2> : [[T:tensor<1024.*]]
    %coeffs1 = arith.constant dense<2> : tensor<1024xi64>
    // CHECK: [[Y:%.+]] = arith.constant dense<3> : [[T]]
    %coeffs2 = arith.constant dense<3> : tensor<1024xi64>
    // CHECK-NOT: poly.from_coeffs
    %poly0 = poly.from_coeffs(%coeffs1) : (tensor<1024xi64>) -> !poly.poly<#ring>
    %poly1 = poly.from_coeffs(%coeffs2) : (tensor<1024xi64>) -> !poly.poly<#ring>
    // CHECK: [[XEXT:%.+]] = arith.extui [[X]] : [[T]] to [[TEXT:tensor<1024.*]]
    // CHECK-NEXT: [[YEXT:%.+]] = arith.extui [[Y]] : [[T]] to [[TEXT:tensor<1024.*]]
    // CHECK-NEXT: [[ZEXT:%.+]] = arith.addi [[XEXT]], [[YEXT]] : [[TEXT]]
    // CHECK-NEXT:  [[MOD:%.+]] = arith.constant dense<4294967296> : [[TEXT]]
    // CHECK-NEXT: [[CMP:%.+]] = arith.cmpi uge, [[ZEXT]], [[MOD]] : [[TEXT]]
    // CHECK-NEXT: [[SUB:%.+]] = arith.subi [[ZEXT]], [[MOD]] : [[TEXT]]
    // CHECK-NEXT: [[SEL:%.+]] = arith.select [[CMP]], [[SUB]], [[ZEXT]] : tensor<1024xi1>, [[TEXT]]
    // CHECK-NEXT:  [[Z:%.+]] = arith.trunci [[SEL]] : [[TEXT]] to [[T]]
    %poly2 = poly.add(%poly0, %poly1) {ring = #ring} : !poly.poly<#ring>
    // CHECK: return  [[Z]] : [[T]]
    return %poly2 : !poly.poly<#ring>
  }
}
