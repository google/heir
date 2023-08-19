// RUN: heir-opt --bgv-to-poly %s > %t
// RUN: FileCheck %s < %t

// This simply tests for syntax.

#my_poly = #poly.polynomial<1 + x**1024>
#ring1 = #poly.ring<cmod=463187969, ideal=#my_poly>
#ring2 = #poly.ring<cmod=33538049, ideal=#my_poly>
#rings = #bgv.rings<#ring1, #ring2>

// CHECK: module
module {
  // CHECK: func.func @test_fn([[X:%.+]]: [[T:tensor<2x!poly.*33538049.*]]) -> [[T]] {
  func.func @test_fn(%x : !bgv.ciphertext<rings=#rings, dim=2, level=1>) -> !bgv.ciphertext<rings=#rings, dim=2, level=1> {
    // CHECK: return [[X]] : [[T]]
    return %x : !bgv.ciphertext<rings=#rings, dim=2, level=1>
  }
}
