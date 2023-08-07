// RUN: heir-opt %s > %t
// RUN: FileCheck %s < %t

// This simply tests for syntax.

#my_poly = #poly.polynomial<1 + x**1024>
#ring1 = #poly.ring<cmod=463187969, ideal=#my_poly>
#ring2 = #poly.ring<cmod=33538049, ideal=#my_poly>
#rings = #bgv.rings<#ring1, #ring2>

// CHECK: module
module {
  func.func @test_multiply(%arg0 : !bgv.ciphertext<rings=#rings>, %arg1: !bgv.ciphertext<rings=#rings>) -> !bgv.ciphertext<rings=#rings> {
    %0 = bgv.mul(%arg0, %arg1) : !bgv.ciphertext<rings=#rings>, !bgv.ciphertext<rings=#rings> -> !bgv.ciphertext<rings=#rings, dim=3>
    %1 = bgv.relinearize(%0) {from_basis = array<i32: 0, 1, 2>, to_basis = array<i32: 0, 1> } : (!bgv.ciphertext<rings=#rings, dim=3>) -> !bgv.ciphertext<rings=#rings>
    %2 = bgv.modulus_switch(%1) {from_level = 1, to_level=0} : !bgv.ciphertext<rings=#rings>
    // CHECK: <<cmod=463187969, ideal=#poly.polynomial<1 + x**1024>>, <cmod=33538049, ideal=#poly.polynomial<1 + x**1024>>>
    return %arg0 : !bgv.ciphertext<rings=#rings>
  }
}
