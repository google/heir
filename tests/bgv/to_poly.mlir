// RUN: heir-opt --bgv-to-poly %s > %t
// RUN: FileCheck %s < %t

// This simply tests for syntax.

#my_poly = #poly.polynomial<1 + x**1024>
#ring1 = #poly.ring<cmod=463187969, ideal=#my_poly>
#ring2 = #poly.ring<cmod=33538049, ideal=#my_poly>
#rings = #bgv.rings<#ring1, #ring2>
!ct1 = !bgv.ciphertext<rings=#rings, dim=2, level=1>

// CHECK: module
module {
  // CHECK: func.func @test_fn([[X:%.+]]: [[T:tensor<2x!poly.*33538049.*]]) -> [[T]] {
  func.func @test_fn(%x : !ct1) -> !ct1 {
    // CHECK: return [[X]] : [[T]]
    return %x : !ct1
  }


  // CHECK: func.func @test_bin_ops([[X:%.+]]: [[T:tensor<2x!poly.*33538049.*]], [[Y:%.+]]: [[T]]) {
  func.func @test_bin_ops(%x : !ct1, %y : !ct1) {
    // CHECK: poly.add([[X]], [[Y]]) : [[T]]
    %add = bgv.add(%x, %y) : !ct1
    // CHECK: poly.sub([[X]], [[Y]]) : [[T]]
    %sub = bgv.sub(%x, %y) : !ct1
    // CHECK: [[C:%.+]] = arith.constant -1 : [[I:.+]]
    // CHECK: poly.mul_constant([[X]], [[C]]) : ([[T]], [[I]]) -> [[T]]
    %negate = bgv.negate(%x) : (!ct1) -> !ct1
    return
  }
}
